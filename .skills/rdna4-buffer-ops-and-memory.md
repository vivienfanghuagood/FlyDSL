# RDNA4 Buffer Operations and Memory Access Patterns

These patterns are used across all WMMA kernels. Key examples:
- GEMM cooperative loading: `kernels/wmma_preshuffle_gemm.py`, `kernels/wmma_gemm.py`
- Preshuffle direct loads: `kernels/wmma_preshuffle_gemm.py`, `kernels/wmma_mixed_preshuffle_gemm.py`
- FP8 dwordx2 loads: `kernels/wmma_mixed_preshuffle_gemm.py`
- INT4 nibble extraction: `kernels/wmma_w4a16_gemv.py`
- Paged KV cache access: `kernels/wmma_decode_attention.py`

## Buffer Resource Descriptors

AMD GPUs use **buffer resource descriptors** for efficient memory access.
A descriptor is a 128-bit (4-DWORD) structure stored in SGPRs containing:
- Base pointer (48-bit virtual address)
- Stride (for structured buffers, usually 0)
- num_records (buffer size in bytes for OOB checking)
- Flags (data format, OOB select)

### Creating a Buffer Resource

```python
from flydsl.dialects.ext import buffer_ops

# From a memref value (most common)
rsrc = buffer_ops.create_buffer_resource(memref_val, max_size=True)

# With specific size for hardware OOB checking
rsrc = buffer_ops.create_buffer_resource(
    memref_val, max_size=False,
    num_records_bytes=arith.constant(4, type=i32)
)
```

### RDNA vs CDNA Flag Differences

```python
# RDNA (gfx10xx, gfx11xx, gfx12xx):
#   OOB_SELECT=2 (disable OOB check) at bits[29:28]
#   FORMAT bit 13 (raw 32-bit)
flags = (2 << 28) | (1 << 13)

# CDNA (gfx9xx):
#   data_format=7 (float) at bits[14:12]
#   num_format=4 (32-bit) at bits[17:15]
flags = (7 << 12) | (4 << 15)
```

## Buffer Load Patterns

### Basic Vector Load

```python
# Load 4xf32 (128 bits = buffer_load_b128)
data = buffer_ops.buffer_load(rsrc, offset, vec_width=4, dtype=ir.F32Type.get())

# Load 8xbf16 (128 bits via 4xf32 then bitcast)
raw = buffer_ops.buffer_load(rsrc, f32_offset, vec_width=4, dtype=ir.F32Type.get())
bf16_vec = vector.bitcast(v8bf16_ty, raw)

# Load 8xbf16 directly
data = buffer_ops.buffer_load(rsrc, offset, vec_width=8, dtype=ir.BF16Type.get())

# Load scalar
val = buffer_ops.buffer_load(rsrc, offset, vec_width=1, dtype=ir.F32Type.get())
```

**Critical**: The `offset` parameter is in **elements**, not bytes. The API internally
multiplies by element size. For bf16 loads via f32 type, divide element offset by 2:

```python
# bf16 element offset -> f32 dword offset for buffer_load
elem_off = row * K + col     # offset in bf16 elements
f32_off = elem_off // 2      # offset in f32 dwords
raw = buffer_ops.buffer_load(rsrc, f32_off, vec_width=4, dtype=f32)
vec_bf16 = vector.bitcast(v8bf16_ty, raw)
```

### Load with dwordx2 (for FP8)

```python
# Load 8 bytes as 2xi32 (for fp8 data)
v2i32 = buffer_ops.buffer_load(rsrc, dword_off, vec_width=2, dtype=i32)
```

### Predicated Load

```python
# Load with mask -- invalid lanes get offset 0x7FFFFFFF (OOB, returns 0)
data = buffer_ops.buffer_load(rsrc, offset, vec_width=4, mask=valid_mask)
```

## Buffer Store Patterns

```python
# Store scalar
buffer_ops.buffer_store(val, rsrc, elem_offset)

# Store with mask
buffer_ops.buffer_store(val, rsrc, elem_offset, mask=valid_mask)

# Store with byte offset (skip element-to-byte conversion)
buffer_ops.buffer_store(val, rsrc, byte_off, offset_is_bytes=True)
```

## LDS (Local Data Share) Operations

### Allocation via SmemAllocator

```python
from flydsl.utils import SmemAllocator

allocator = SmemAllocator(None, arch=gpu_arch)

# Allocate arrays
lds_a = allocator.allocate_array(bf16_ty, num_elements)
lds_b = allocator.allocate_array(bf16_ty, num_elements)
allocator.finalize()

# In kernel: get view
lds_base = allocator.get_base()
a_view = lds_a(lds_base).get()  # flat 1D memref view
```

### LDS Vector Operations

```python
# Store vector to LDS
vector.store(v8bf16_vec, lds_view, [lds_index])

# Load vector from LDS
v8bf16 = vector.load_op(v8bf16_ty, lds_view, [lds_index])
```

### LDS Bank Conflict Avoidance

RDNA4 has 32 LDS banks, each 4 bytes wide (128 bytes per bank cycle).

**K-padding strategy**: Add 8 elements padding per row to shift bank access patterns.

```python
BLOCK_K_PAD = BLOCK_K + 8  # 32 + 8 = 40 elements per row
lds_idx = row * BLOCK_K_PAD + col
```

**XOR-swizzle strategy**: XOR the K-group index with row index:

```python
# For LOAD_VEC=8 bf16 per store, 4 K-groups per row
# k_group_swizzled = k_group XOR (row % 4)
# Ensures adjacent rows use different bank patterns
```

### LDS Inline ASM (Advanced)

For precise control over LDS read scheduling:

```python
# ds_load_u16: Load 16-bit scalar from LDS
# ds_load_u16_d16_hi: Load 16-bit and place in high half of 32-bit register
# ds_load_b128: Load 128 bits (4 dwords) from LDS

# Combined pattern with embedded waits:
asm = """
ds_load_u16 $0, $base offset:0
ds_load_u16 $1, $base offset:16
...
ds_load_b128 $16, $a_base offset:0
s_wait_dscnt 0x4           // wait for b128 loads only
ds_load_u16_d16_hi $0, $base offset:544
...
s_wait_dscnt 0x0           // wait for everything
"""
```

## Memory Coalescing on RDNA4

### Optimal Access Patterns

1. **Contiguous**: Adjacent threads access adjacent memory locations (best)
2. **Strided**: Threads access memory with constant stride (acceptable if stride < 32)
3. **Scattered**: Random access (worst, avoid for GMEM)

### For GEMM A[M,K] Row-Major Loading

```python
# Each thread loads 8 contiguous bf16 elements
# Thread layout: linearize all 128 threads
a_lin = tid * 8 + load_idx * THREADS_PER_BLOCK * 8
a_row = a_lin // BLOCK_K   # which M-row
a_col = a_lin % BLOCK_K    # which K-column

# This gives coalesced access: threads 0-3 load row 0, threads 4-7 load row 1, etc.
# Each thread's load is contiguous along K (8 bf16 = 16 bytes)
```

### For Preshuffle B Access

When B is preshuffled, each thread loads its own WMMA operand from a unique,
contiguous 16-byte region. Access is inherently coalesced because:
- `lane16` varies across threads (column dimension)
- `klane` varies across thread halves (K dimension)
- `KPack=8` elements are contiguous per thread

## Output Store Patterns

### Scalar Stores (Simple)

```python
for si in range(8):   # 8 results per accumulator vector
    g_row = tile_m0 + wmma_m_off + klane * 8 + si
    g_col = tile_n0 + wmma_n_off + lane16
    val = vector.extract(acc, static_position=[si])
    if out_dtype == "bf16":
        val = arith.trunc_f(bf16_ty, val)
    buffer_ops.buffer_store(val, c_rsrc, g_row * N + g_col)
```

### Vectorized Stores (Better)

For bf16 output where 2 values can be packed:

```python
# Pack 2 bf16 values into 1 i32 for dword store
# (Not shown in current kernels but achievable)
```

## Wave-Uniform vs Per-Thread Values

- **Wave-uniform** (SGPR): Values identical across all threads in a wave
  - Block ID, tile base addresses, K-tile index
  - Use for buffer resource base pointers

- **Per-thread** (VGPR): Values differ per thread
  - lane16, klane, thread-local offsets
  - WMMA operand data, accumulator results
  - Use for buffer load offsets (each thread loads different data)
