# FlyDSL API Dictionary

Comprehensive reference for all FlyDSL functions, classes, and operations used to develop GPU kernels.
Organized by module. All imports are from `flydsl.dialects.ext.*` unless noted otherwise.

---

## Table of Contents

1. [Module & Kernel Decorators (flir)](#1-module--kernel-decorators-flir)
2. [Arithmetic (arith)](#2-arithmetic-arith)
3. [Buffer Operations (buffer_ops)](#3-buffer-operations-buffer_ops)
4. [ROCDL Intrinsics (rocdl)](#4-rocdl-intrinsics-rocdl)
5. [Vector Operations (vector)](#5-vector-operations-vector)
6. [Structured Control Flow (scf)](#6-structured-control-flow-scf)
7. [GPU Dialect (gpu)](#7-gpu-dialect-gpu)
8. [MemRef Operations (memref)](#8-memref-operations-memref)
9. [LLVM Dialect (llvm)](#9-llvm-dialect-llvm)
10. [Math Operations (math)](#10-math-operations-math)
11. [Block Reduce Operations (block_reduce_ops)](#11-block-reduce-operations-block_reduce_ops)
12. [Shared Memory Allocator (SmemAllocator)](#12-shared-memory-allocator-smemallocator)
13. [Type System (Types / T)](#13-type-system-types--t)
14. [Compiler & Execution](#14-compiler--execution)
15. [Python Control Flow (python_control_flow)](#15-python-control-flow-python_control_flow)
16. [ROCm Helpers (rocm)](#16-rocm-helpers-rocm)

---

## 1. Module & Kernel Decorators (flir)

**Import**: `from flydsl.dialects.ext import flir`

### flir.MlirModule (base class)

Base class for defining a GPU module containing kernels.

```python
class MyKernel(flir.MlirModule):
    GPU_MODULE_NAME = "my_kernel"                           # Required: module symbol name
    GPU_MODULE_TARGETS = ['#rocdl.target<chip = "gfx1201">']  # Required: GPU target

    def init_gpu_module(self):
        """Optional: Called during module construction. Use for LDS allocation."""
        self._smem = allocator.allocate_array(T.f32(), 256)
        allocator.finalize()

    @flir.kernel
    def kernel_func(self, arg: lambda: T.memref(DYN, T.bf16()), ...):
        """GPU kernel function. Emitted as gpu.func @kernel_func kernel."""
        ...

    @flir.jit
    def __call__(self, ..., stream_ptr: lambda: I.i64):
        """Host-side launcher. Emitted as func.func @__call__."""
        ...
```

### @flir.kernel

Decorator for GPU kernel functions. Emits `gpu.func @name kernel { ... }`.
- Automatically lowers Python `for i in range(...)` into `scf.for`
- Function arguments use lambda type annotations: `arg: lambda: T.memref(DYN, T.bf16())`
- First argument `self` must be typed as `flir.T.i64` (module handle)

### @flir.jit

Decorator for host-side JIT functions. Emits `func.func @name { ... }`.
- Used for `__call__` to create the launch function
- Contains `LaunchFuncOp` to dispatch the kernel

### Thread/Block ID helpers

```python
tid_x = flir.thread_idx("x")   # gpu.thread_id "x" -> index
tid_y = flir.thread_idx("y")
pid_x = flir.block_idx("x")    # gpu.block_id "x" -> index
pid_y = flir.block_idx("y")
pid_z = flir.block_idx("z")
```

### gpu.LaunchFuncOp (from flir.gpu_ext)

```python
flir.gpu_ext.LaunchFuncOp(
    ["module_name", "kernel_func_name"],          # kernel reference
    grid_size=(grid_x, grid_y, grid_z),           # index-typed values or ints
    block_size=(block_x, block_y, block_z),       # index-typed values or ints
    kernel_operands=[arg1, arg2, ...],             # memrefs and scalars
    async_dependencies=[stream_token],             # optional async token
    dynamic_shared_memory_size=smem_bytes,         # optional index value
)
```

---

## 2. Arithmetic (arith)

**Import**: `from flydsl.dialects.ext import arith`

### Constants

| Function | Signature | Description |
|---|---|---|
| `arith.constant(value, *, type=None, index=False)` | `-> ArithValue` | Create typed constant |
| `arith.index(value)` | `-> ArithValue` | Create index-type constant |
| `arith.i32(value)` | `-> ArithValue` | Create i32 constant |
| `arith.i64(value)` | `-> ArithValue` | Create i64 constant |
| `arith.f16(value)` | `-> ArithValue` | Create f16 constant |
| `arith.f32(value)` | `-> ArithValue` | Create f32 constant |
| `arith.f64(value)` | `-> ArithValue` | Create f64 constant |
| `arith.constant_vector(elem_val, vec_type)` | `-> ArithValue` | Create splat vector (all elements same) |

```python
c0 = arith.index(0)
c42 = arith.i32(42)
pi = arith.f32(3.14159)
zero_vec = arith.constant_vector(0.0, ir.VectorType.get([8], ir.F32Type.get()))
```

### Type Conversions

| Function | Signature | Description |
|---|---|---|
| `arith.extf(result_type, value)` | `-> ArithValue` | Widen float (f16->f32, bf16->f32) |
| `arith.trunc_f(target_type, value)` | `-> ArithValue` | Narrow float (f32->f16, f32->bf16) |
| `arith.fptosi(result_type, value)` | `-> ArithValue` | Float to signed integer |
| `arith.sitofp(result_type, value)` | `-> ArithValue` | Signed integer to float |
| `arith.uitofp(result_type, value)` | `-> ArithValue` | Unsigned integer to float |
| `arith.index_cast(target_type, value)` | `-> ArithValue` | Cast between index and integer |
| `arith.index_cast_ui(target_type, value)` | `-> ArithValue` | Unsigned cast between index and integer |
| `arith.bitcast(result_type, value)` | `-> ArithValue` | Reinterpret bits (same width) |

```python
f32_val = arith.extf(ir.F32Type.get(), bf16_val)
bf16_val = arith.trunc_f(ir.BF16Type.get(), f32_val)
i32_val = arith.index_cast(ir.IntegerType.get_signless(32), idx_val)
i16_vec = arith.bitcast(v8i16_ty, bf16_vec)  # bf16 -> i16 for WMMA
```

### Bitwise Operations

| Function | Signature | Description |
|---|---|---|
| `arith.andi(lhs, rhs)` | `-> ArithValue` | Bitwise AND |
| `arith.ori(lhs, rhs)` | `-> ArithValue` | Bitwise OR |
| `arith.xori(lhs, rhs)` | `-> ArithValue` | Bitwise XOR |
| `arith.shrui(lhs, rhs)` | `-> ArithValue` | Logical (unsigned) right shift |
| `arith.shli(lhs, rhs)` | `-> ArithValue` | Left shift |

All accept `ArithValue`, `Value`, or Python `int` arguments.

### Comparisons

| Function | Signature | Description |
|---|---|---|
| `arith.cmpu(lhs, rhs, predicate)` | `-> ArithValue` | Unsigned integer compare |
| `arith.ult(lhs, rhs)` | `-> ArithValue` | Unsigned less than |
| `arith.ule(lhs, rhs)` | `-> ArithValue` | Unsigned less or equal |
| `arith.ugt(lhs, rhs)` | `-> ArithValue` | Unsigned greater than |
| `arith.uge(lhs, rhs)` | `-> ArithValue` | Unsigned greater or equal |

Predicate strings for `cmpu`: `"ult"`, `"ule"`, `"ugt"`, `"uge"`

**Signed comparisons** use Python operators on `ArithValue`:
```python
cond = a < b    # slt (signed)
cond = a == b   # eq
cond = a >= b   # sge (signed)
```

### Min/Max/Select

| Function | Signature | Description |
|---|---|---|
| `arith.maximum(lhs, rhs)` | `-> ArithValue` | Max (float: MaximumFOp, int: MaxSIOp) |
| `arith.minimum(lhs, rhs)` | `-> ArithValue` | Min (float: MinimumFOp, int: MinSIOp) |
| `arith.select(cond, true_val, false_val)` | `-> ArithValue` | Ternary select (like `?:`) |
| `arith.absf(value)` | `-> ArithValue` | Absolute value (float) |

### Vector Reduction

```python
arith.reduce(vec_value, kind="add")   # Sum all vector elements
arith.reduce(vec_value, kind="max")   # Max of all elements
# Kinds: "add", "mul", "min", "max", "and", "or", "xor"
# Optional acc= parameter for accumulator
```

### Value Wrapping/Unwrapping

| Function | Description |
|---|---|
| `arith.unwrap(val)` | Unwrap ArithValue to raw `ir.Value` (for MLIR API calls) |
| `arith.as_value(val)` | Alias for `unwrap` |
| `arith.ArithValue(raw_value)` | Wrap raw `ir.Value` for operator overloading |

### ArithValue Operator Overloading

`ArithValue` supports Python operators that emit MLIR ops:

```python
c = a + b     # AddI/AddF
c = a - b     # SubI/SubF
c = a * b     # MulI/MulF
c = a / b     # DivSI/DivF
c = a % b     # RemSI/RemF
c = a & b     # AndI
c = a | b     # OrI
c = a ^ b     # XOrI
c = a << b    # ShLI
c = a >> b    # ShRUI (unsigned right shift)
c = a == b    # CmpI eq / CmpF OEQ
c = a < b     # CmpI slt / CmpF OLT
c = a.max(b)  # MaximumF / MaxSI
c = a.min(b)  # MinimumF / MinSI
```

Python scalars auto-promote: `a + 3` creates a constant matching `a`'s type.

---

## 3. Buffer Operations (buffer_ops)

**Import**: `from flydsl.dialects.ext import buffer_ops`

AMD buffer load/store operations using hardware buffer resource descriptors.

### Buffer Resource Creation

```python
buffer_ops.create_buffer_resource(
    memref_val,                     # memref value (the tensor)
    stride=0,                       # buffer stride (0 for contiguous)
    max_size=True,                  # use max buffer size (0xFFFFFFFF)
    num_records_bytes=None,         # optional: exact buffer size in bytes
) -> ir.Value                       # returns !llvm.ptr<8> descriptor
```

**Architecture-aware flags**: Automatically detects RDNA vs CDNA and sets appropriate DWORD3 flags:
- RDNA (gfx10/11/12): `OOB_SELECT=2, FORMAT bit 13`
- CDNA (gfx9xx): `data_format=7, num_format=4`

### Buffer Load

```python
buffer_ops.buffer_load(
    rsrc,                           # buffer resource descriptor
    offset,                         # element offset (auto-converted to bytes internally)
    vec_width=4,                    # 1, 2, or 4 elements
    dtype=None,                     # element type (default: f32)
    mask=None,                      # optional i1 predicate (invalid -> OOB offset)
    cache_modifier=0,               # cache control flags
    soffset_bytes=None,             # optional scalar byte offset (folds into instruction)
) -> ir.Value                       # vector or scalar result
```

**Important**: The `offset` parameter is in **elements**. The API multiplies by `dtype.width // 8` internally.

### Buffer Store

```python
buffer_ops.buffer_store(
    data,                           # data to store (scalar or vector)
    rsrc,                           # buffer resource descriptor
    offset,                         # element offset
    mask=None,                      # optional i1 predicate
    cache_modifier=0,               # cache control flags
    soffset_bytes=None,             # optional scalar byte offset
    offset_is_bytes=False,          # True to skip element-to-byte scaling
)
```

### 2D Convenience Wrappers

```python
buffer_ops.buffer_load_2d(rsrc, row, col, stride, vec_width=4, dtype=None, mask=None)
buffer_ops.buffer_store_2d(data, rsrc, row, col, stride, mask=None)
# offset = row * stride + col (computed internally)
```

### Helper Functions

```python
buffer_ops.index_cast_to_i32(value)          # Cast index -> i32
buffer_ops.i32_mul(lhs, rhs)                 # i32 multiply
buffer_ops.i32_add(lhs, rhs)                 # i32 add
buffer_ops.i32_select(cond, true_val, false_val)  # i32 select
```

### LLVM Pointer Helpers

```python
buffer_ops.create_llvm_ptr(index_value, address_space=0)
# Convert index value to LLVM pointer. address_space: 0=generic, 3=LDS, 8=buffer

buffer_ops.get_element_ptr(
    base_ptr,                       # LLVM pointer
    byte_offset=None,               # dynamic byte offset (Value or int)
    static_byte_offset=0,           # constant byte offset
    elem_type=None,                 # GEP element type (default: i8)
) -> ir.Value                       # pointer with offset applied
```

---

## 4. ROCDL Intrinsics (rocdl)

**Import**: `from flydsl.dialects.ext import rocdl`

AMD-specific GPU intrinsics. This module re-exports everything from `_mlir.dialects.rocdl` plus wrapped versions.

### Thread/Block/Grid IDs

```python
rocdl.workitem_id_x()      # Lane ID within workgroup (0..BLOCK_SIZE-1)
rocdl.workitem_id_y()
rocdl.workitem_id_z()
rocdl.workgroup_id_x()     # Block/workgroup ID
rocdl.workgroup_id_y()
rocdl.workgroup_id_z()
rocdl.workgroup_dim_x()    # Workgroup dimensions
rocdl.wavefrontsize()      # 32 for RDNA, 64 for CDNA
```

### Synchronization

```python
rocdl.barrier()                     # Full workgroup barrier
rocdl.s_barrier()                   # Scalar barrier
rocdl.s_barrier_signal(count)       # Signal barrier with count
rocdl.s_barrier_wait(count)         # Wait for barrier signal
rocdl.s_waitcnt(flags)              # Wait for memory operations
rocdl.s_wait_loadcnt(count)         # Wait for N loads to complete
rocdl.s_wait_storecnt(count)        # Wait for N stores to complete
rocdl.s_wait_dscnt(count)           # Wait for N DS (LDS) ops to complete
rocdl.s_wait_expcnt(count)          # Wait for N export ops to complete
```

### WMMA Instructions (RDNA3/RDNA4)

All WMMA ops accept `(result_type, operands_list)` and return the result `ir.Value` directly.

**Float output variants** (operands = `[A, B, C]`):

| Function | A dtype | B dtype | Acc dtype | Notes |
|---|---|---|---|---|
| `rocdl.wmma_f32_16x16x16_f16(v8f32, [A, B, C])` | f16 | f16 | f32 | |
| `rocdl.wmma_f32_16x16x16_bf16(v8f32, [A, B, C])` | bf16 | bf16 | f32 | A,B must be bitcast to v8i16 |
| `rocdl.wmma_f32_16x16x16_fp8_fp8(v8f32, [A, B, C])` | fp8 | fp8 | f32 | gfx12 only, A,B are v2i32 |
| `rocdl.wmma_f32_16x16x16_fp8_bf8(v8f32, [A, B, C])` | fp8 | bf8 | f32 | gfx12 only |
| `rocdl.wmma_f32_16x16x16_bf8_fp8(v8f32, [A, B, C])` | bf8 | fp8 | f32 | gfx12 only |
| `rocdl.wmma_f32_16x16x16_bf8_bf8(v8f32, [A, B, C])` | bf8 | bf8 | f32 | gfx12 only |

**Half output variants** (operands = `[A, B, C, op_sel]`):

| Function | Notes |
|---|---|
| `rocdl.wmma_f16_16x16x16_f16(v8f16, [A, B, C, op_sel])` | op_sel: bool, selects high/low half |
| `rocdl.wmma_bf16_16x16x16_bf16(v8bf16, [A, B, C, op_sel])` | op_sel: bool |

**Integer variants** (operands = `[A_sign, A, B_sign, B, C, clamp]`):

| Function | Notes |
|---|---|
| `rocdl.wmma_i32_16x16x16_iu8(v8i32, [A_sign, A, B_sign, B, C, clamp])` | int8 |
| `rocdl.wmma_i32_16x16x16_iu4(v8i32, [A_sign, A, B_sign, B, C, clamp])` | int4 |
| `rocdl.wmma_i32_16x16x32_iu4(v8i32, [A_sign, A, B_sign, B, C, clamp])` | int4, gfx12 only |

All have `_op` variants (e.g., `wmma_f32_16x16x16_bf16_op(...)`) that return the op view instead of `.result`.

### MFMA Instructions (CDNA)

```python
rocdl.mfma_f32_16x16x16f16(v4f32, [A, B, C, cbsz, abid, blgp])
rocdl.mfma_f32_16x16x16bf16_1k(v4f32, [A, B, C, cbsz, abid, blgp])
rocdl.mfma_f32_16x16x32_fp8_fp8(v4f32, [A, B, C, cbsz, abid, blgp])
rocdl.mfma_i32_16x16x32_i8(v4i32, [A, B, C, cbsz, abid, blgp])
rocdl.mfma_scale_f32_16x16x128_f8f6f4(v4f32, [A, B, C, ...])  # gfx950 only
```

### Shuffle / Cross-Lane Operations

```python
rocdl.readlane(result_type, src, lane_id)      # Read from specific lane
rocdl.readfirstlane(result_type, src)           # Read from first active lane
rocdl.ds_swizzle(result_type, src, offset)      # DS permute within wave
rocdl.ds_bpermute(result_type, src, offset)     # Byte permute across wave
rocdl.permlanex16(...)                          # RDNA cross-lane permute
rocdl.permlane16_swap(...)                      # RDNA lane swap (16-lane halves)
rocdl.permlane32_swap(...)                      # RDNA lane swap (32-lane)
rocdl.update_dpp(...)                           # DPP (Data-Parallel Primitive) update
rocdl.ballot(...)                               # Active lane ballot
```

### Buffer Load/Store (raw ROCDL level)

```python
rocdl.raw_ptr_buffer_load(result_type, rsrc, voffset, soffset, aux)
rocdl.raw_ptr_buffer_store(data, rsrc, voffset, soffset, aux)
rocdl.load_to_lds(...)                          # Direct GMEM->LDS load
rocdl.global_load_lds(...)                      # Global load to LDS
rocdl.make_buffer_rsrc(rsrc_type, base_ptr, stride, num_records, flags)
```

### Atomic Operations

```python
rocdl.raw_ptr_buffer_atomic_fadd(val, rsrc, voffset, soffset, cache)
rocdl.raw_ptr_buffer_atomic_fmax(...)
rocdl.raw_buffer_atomic_fadd(...)
rocdl.raw_buffer_atomic_smax(...)
```

### Scheduling Hints

```python
rocdl.sched_barrier(mask)                       # Instruction scheduling barrier
rocdl.sched_group_barrier(mask, count, syncid)  # Group scheduling barrier
rocdl.iglp_opt(mode)                            # IGLP optimization hint
rocdl.s_setprio(priority)                       # Set instruction priority
rocdl.s_sleep(count)                            # Sleep N cycles

# Convenience scheduling helpers:
rocdl.sched_mfma(cnt)      # Schedule cnt MFMA instructions (mask=0x008)
rocdl.sched_vmem(cnt)      # Schedule cnt VMEM reads (mask=0x020)
rocdl.sched_dsrd(cnt)      # Schedule cnt DS reads (mask=0x100)
rocdl.sched_dswr(cnt)      # Schedule cnt DS writes (mask=0x200)
```

### Type Conversions

```python
rocdl.cvt_f32_bf8(...)             # BF8 -> F32
rocdl.cvt_f32_fp8(...)             # FP8 -> F32
rocdl.cvt_pk_f32_bf8(...)          # Packed BF8 -> 2xF32
rocdl.cvt_pk_f32_fp8(...)          # Packed FP8 -> 2xF32
```

### Bit Manipulation

```python
rocdl.mbcnt_lo(mask, init)         # Count bits in lower 32 lanes
rocdl.mbcnt_hi(mask, init)         # Count bits in upper 32 lanes
```

---

## 5. Vector Operations (vector)

**Import**: `from flydsl.dialects.ext import vector`

Re-exports everything from `_mlir.dialects.vector` plus wrapped versions.

### Core Operations

```python
vector.extract(vec, static_position=[i], dynamic_position=[])
# Extract element at static index i from vector. Returns scalar Value.

vector.load_op(result_type, memref, indices)
# Load vector from memref at given indices. Returns vector Value.

vector.load(memref, indices)
# Alternative load (auto-infers result type from memref).

vector.store(value, memref, indices)
# Store vector to memref at given indices.

vector.bitcast(result_type, source)
# Reinterpret vector bits as different element type.
# Example: v8bf16 -> v8i16 (required before WMMA)

vector.shuffle(v1, v2, mask)
# Shuffle elements from two vectors using mask. Returns vector Value.

vector.broadcast(result_type, source)
# Broadcast scalar or smaller vector to larger vector type.

vector.from_elements(result_type, [e0, e1, ...])
# Build vector from individual scalar elements.

vector.transfer_read(result_type, source, indices, permutation_map, padding, in_bounds)
# Advanced: multi-dimensional vector transfer read.
```

### Re-exported from _mlir.dialects.vector

All standard MLIR vector ops are available, including:
- `vector.InsertOp`, `vector.ExtractOp`
- `vector.BroadcastOp`, `vector.ShuffleOp`, `vector.BitCastOp`
- `vector.TransferReadOp`, `vector.TransferWriteOp`
- `vector.FMAOp`, `vector.SplatOp`
- `vector.ReductionOp` (prefer `arith.reduce()` for convenience)
- `vector.CombiningKind` enum

---

## 6. Structured Control Flow (scf)

**Import**: `from flydsl.dialects.ext import scf`

### For Loop (context manager)

```python
# Simple loop (Python range semantics)
with scf.range_(stop) as i:
    # i is an index-type induction variable
    ...

with scf.range_(start, stop, step) as i:
    ...

# Loop with carried values (loop-carried variables)
with scf.range_(10, iter_args=[init_val]) as (i, val):
    # val is the loop-carried value
    new_val = val + arith.index(1)
    scf.yield_([new_val])

# Negative step (static bounds only)
with scf.range_(10, 0, -1) as i:
    ...
```

### For Loop (op access)

```python
# When you need access to the op for .results:
with scf.for_(start, stop, step, iter_args=[init]) as for_op:
    i = for_op.induction_variable
    carried = for_op.inner_iter_args[0]
    scf.yield_([new_carried])
result = for_op.results[0]
```

### If/Else

```python
# Simple if (no results):
_if = scf.IfOp(condition)          # condition must be i1 Value
with _if.then():
    # then body
    scf.yield_([])                  # always terminate with yield

# Full pattern with context managers:
_if = scf.IfOp(condition, hasElse=True)
with _if.then():
    ...
    scf.yield_([])
with _if.else_():
    ...
    scf.yield_([])

# Default context (enters then-block):
with scf.IfOp(condition) as if_op:
    # this is the then block
    ...
```

### Raw scf.IfOp (for kernels)

```python
from _mlir.dialects import scf as raw_scf
if_op = raw_scf.IfOp(condition_value)
with ir.InsertionPoint(if_op.then_block):
    # ... then body ...
    raw_scf.YieldOp([])
```

### Other

```python
scf.yield_(operands=[val1, val2])  # Yield values from region
scf.ForOp(start, stop, step, iter_args)  # Wrapper that accepts ints
scf.WhileOp, scf.YieldOp, scf.ExecuteRegionOp  # Re-exported from MLIR
```

---

## 7. GPU Dialect (gpu)

**Import**: `from flydsl.dialects.ext import gpu`

### Thread/Block IDs

```python
gpu.thread_id("x")          # -> index (0..BLOCK_SIZE-1)
gpu.thread_id("y")
gpu.block_id("x")           # -> index (block/workgroup ID)
gpu.block_id("y")
gpu.block_dim("x")          # -> index (block dimension)
gpu.grid_dim("x")           # -> index (grid dimension)
```

### Synchronization

```python
gpu.barrier()                # gpu.barrier (workgroup barrier)
# WARNING: gpu.barrier() inserts buffer_gl_inv on RDNA4. Use inline asm barrier to avoid it.
```

### Memory Allocation

```python
gpu.alloc(sizes, element_type, host_shared=None)
gpu.dealloc(memref)
gpu.memcpy(dst, src)
gpu.dynamic_shared_memory(int=False)   # Returns dynamic shared memory memref
```

### Module Construction

```python
gpu.GPUModuleOp(sym_name, targets=[...])    # Create gpu.module
gpu.GPUFuncOp(sym_name, function_type)       # Create gpu.func
gpu.LaunchFuncOp(kernel, grid_size, block_size, kernel_operands, ...)
gpu.LaunchOp(grid_size, block_size, ...)     # Inline gpu.launch
```

### GPU Attributes

```python
gpu.smem_space()             # #gpu.address_space<workgroup> attribute
gpu.lds_space()              # Alias for smem_space()
gpu.smem_space(int=True)     # Returns integer value of workgroup address space
```

### Shuffle (Cross-Lane)

```python
# Must use raw MLIR values (_unwrap before passing):
shuf = gpu.ShuffleOp(
    value,                   # raw Value (must be i32 or f32)
    offset,                  # raw Value (i32)
    width,                   # raw Value (i32, typically 32 for wave32)
    mode="xor"               # "xor", "up", "down", "idx"
)
result = shuf.shuffleResult  # the shuffled value
valid  = shuf.valid          # i1 validity flag
```

### All-Reduce

```python
gpu.all_reduce_(value, op="add")    # Returns reduced value
gpu.all_reduce_(value, op="max")
```

### Other

```python
gpu.printf(format_str, *args)              # GPU printf for debugging
gpu.func(f, emit=True, ...)                # Decorator for GPU functions
gpu.kernel(f, ...)                         # Decorator for GPU kernel functions
gpu.ReturnOp(...)                          # Return from GPU function
gpu.TerminatorOp()                         # Terminate launch region
gpu.get_compile_object_bytes(module)       # Extract compiled binary
```

---

## 8. MemRef Operations (memref)

**Import**: `from flydsl.dialects.ext import memref`

Re-exports everything from `_mlir.dialects.memref` plus wrapped versions that accept ArithValue.

### Core Operations

```python
memref.load(memref_val, indices)             # Load scalar from memref
memref.store(value, memref_val, indices)     # Store scalar to memref
memref.view(source, byte_shift, sizes)       # Create typed view of raw buffer
memref.get_global(memref_type, sym_name)     # Access a global memref (e.g., LDS)
```

### Re-exported from _mlir.dialects.memref

```python
memref.alloc(memref_type, ...)               # Allocate memref
memref.dealloc(memref_val)                   # Deallocate memref
memref.extract_aligned_pointer_as_index(memref_val)  # Get base pointer as index
memref.subview(...)                          # Create a subview
memref.global_(sym_name, type_, alignment)   # Declare global memref
memref.dim(memref_val, index)                # Get dynamic dimension
memref.cast(target_type, source)             # Cast memref type
```

---

## 9. LLVM Dialect (llvm)

**Import**: `from flydsl.dialects.ext import llvm`

### Intrinsic Calls

```python
llvm.call_intrinsic(
    result_types,                            # list of result types, or single type
    intrin_name,                             # "llvm.amdgcn.exp2.f32", etc.
    operands,                                # list of Values (auto-unwraps ArithValue)
)
```

Common AMD intrinsics:
```python
llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [x])      # Fast 2^x
llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [x])       # Fast 1/x
llvm.call_intrinsic(f32, "llvm.amdgcn.log2.f32", [x])      # Fast log2(x)
llvm.call_intrinsic(f32, "llvm.amdgcn.rsq.f32", [x])       # Fast 1/sqrt(x)
llvm.call_intrinsic(f32, "llvm.amdgcn.fract.f32", [x])     # Fractional part
llvm.call_intrinsic(f32, "llvm.amdgcn.fmed3.f32", [a,b,c]) # Median of 3
```

### Inline Assembly

```python
llvm.inline_asm(
    result_type,                             # None for void, or struct type
    operands,                                # list of input values
    asm_string,                              # assembly text
    constraints,                             # "=&v,=&v,v,v" style
    has_side_effects=True,
    is_align_stack=False,
)
```

Example: LDS load via inline asm:
```python
lds_val = llvm.inline_asm(
    ir.IntegerType.get_signless(32),
    [lds_addr_i32],
    "ds_read_b32 $0, $1",
    "=v,v",
    has_side_effects=True,
)
```

### Re-exported LLVM Operations

```python
llvm.IntToPtrOp(ptr_type, int_value)         # Integer to pointer
llvm.GEPOp(result_type, base, indices, ...)  # GetElementPtr
llvm.LoadOp(result_type, ptr)                # LLVM load
llvm.StoreOp(value, ptr)                     # LLVM store
llvm.BitcastOp(result_type, value)           # LLVM bitcast
llvm.UndefOp(type)                           # Create undef value
llvm.InsertValueOp(...)                      # Insert into struct
llvm.ExtractValueOp(...)                     # Extract from struct
```

---

## 10. Math Operations (math)

**Import**: `from flydsl.dialects.ext import math as flydsl_math`

Re-exports all ops from `_mlir.dialects.math`:

```python
flydsl_math.exp(x)           # e^x (prefers llvm.amdgcn.exp2 for perf)
flydsl_math.exp2(x)          # 2^x (fast on AMD GPUs)
flydsl_math.log(x)           # ln(x)
flydsl_math.log2(x)          # log2(x)
flydsl_math.sqrt(x)          # sqrt(x)
flydsl_math.rsqrt(x)         # 1/sqrt(x)
flydsl_math.sin(x)           # sin(x)
flydsl_math.cos(x)           # cos(x)
flydsl_math.tanh(x)          # tanh(x)
flydsl_math.floor(x)         # floor(x)
flydsl_math.ceil(x)          # ceil(x)
flydsl_math.powf(base, exp)  # base^exp
flydsl_math.fma(a, b, c)     # a*b + c (fused)
flydsl_math.absf(x)          # |x| (prefer arith.absf)
```

**Performance tip**: For softmax, use `exp2(x * LOG2E)` instead of `exp(x)`:
```python
LOG2E = 1.4426950408889634
exp_val = flydsl_math.exp2(arith.as_value(x * arith.constant(LOG2E, type=f32)))
```

---

## 11. Block Reduce Operations (block_reduce_ops)

**Import**: `from flydsl.dialects.ext import block_reduce_ops`

High-level collective reductions for entire thread blocks.

```python
block_reduce_ops.block_reduce_max(val, red_smem, tid, num_warps=4, warp_size=64)
block_reduce_ops.block_reduce_sum(val, red_smem, tid, num_warps=4, warp_size=64)
block_reduce_ops.block_reduce_min(val, red_smem, tid, num_warps=4, warp_size=64)
```

Parameters:
- `val`: Per-thread scalar value (f32)
- `red_smem`: Shared memory buffer for inter-warp reduction (shape >= `[num_warps]`)
- `tid`: Thread ID within block (index type)
- `num_warps`: Number of warps per block
- `warp_size`: 32 for RDNA, 64 for CDNA

Algorithm: warp-level shuffle reduction -> lane-0 writes to smem -> thread-0 final reduction -> broadcast result.

---

## 12. Shared Memory Allocator (SmemAllocator)

**Import**: `from flydsl.utils import SmemAllocator, SmemPtr, SmemStructInstance`

### SmemAllocator

Manages static LDS (shared memory) allocation with automatic alignment.

```python
allocator = SmemAllocator(ctx=None, arch="gfx1201")

# Allocate an array
gen_array = allocator.allocate_array(dtype=T.f32(), num_elems=256, alignment=None)

# Allocate raw bytes
gen_bytes = allocator.allocate(1024)

# Allocate a scalar
gen_scalar = allocator.allocate(T.f32())

# Allocate a tensor with shape
gen_tensor = allocator.allocate_tensor(layout=(16, 16), element_type=T.f32())

# Allocate a struct (Python dataclass with MLIR type annotations)
@dataclass
class MyStruct:
    counter: T.i32
    value: T.f32
gen_struct = allocator.allocate(MyStruct)

# Finalize: emits memref.global in gpu.module
allocator.finalize()

# In kernel: get base pointer
base = allocator.get_base()

# Materialize allocations
array_smem = gen_array(base)        # -> SmemPtr
scalar_smem = gen_scalar(base)      # -> SmemPtr
struct_smem = gen_struct(base)      # -> SmemStructInstance
```

### SmemPtr

Typed pointer into shared memory. Wraps `memref.view` over the raw LDS buffer.

```python
smem_ptr = SmemPtr(base_memref, byte_offset, element_type, shape=(256,))

view = smem_ptr.get()               # -> memref value (creates memref.view)
smem_ptr.load(indices)               # Load from LDS
smem_ptr.store(value, indices)       # Store to LDS
```

### SmemStructInstance

Struct-like access to LDS fields:

```python
struct_inst = gen_struct(base)
counter_ptr = struct_inst.counter    # -> SmemPtr to counter field
counter_ptr.store(arith.i32(0))
val = counter_ptr.load()
```

### LDS Capacity

Known capacities per arch:
- `gfx942` (MI300): 65536 bytes (64KB)
- `gfx950` (MI350): 163840 bytes (160KB)
- `gfx1201` (RDNA4): 65536 bytes (64KB)

Allocation exceeding capacity raises `RuntimeError`.

---

## 13. Type System (Types / T)

**Import**: `from flydsl.lang.ir.types import T as I` (custom types) or `import _mlir.extras.types as T` (MLIR types)

### Custom Type Singleton (I = Types())

Provides property-based access to common types and pre-built vector types:

**Scalars**:
`I.index`, `I.i8`, `I.ui8`, `I.i16`, `I.i32`, `I.ui32`, `I.i64`,
`I.f16`, `I.bf16`, `I.f32`, `I.f64`, `I.f8`, `I.f4`, `I.e8m0`

**Pre-built vectors**:
`I.i8x2`, `I.i8x4`, `I.i8x8`, `I.i8x16`,
`I.ui8x2`, `I.ui8x4`, `I.ui8x8`, `I.ui8x16`,
`I.i16x2`, `I.i16x4`, `I.i16x8`,
`I.i32x2`, `I.i32x4`,
`I.i64x2`,
`I.f16x1`, `I.f16x2`, `I.f16x4`, `I.f16x8`,
`I.bf16x2`, `I.bf16x4`, `I.bf16x8`,
`I.f32x2`, `I.f32x4`,
`I.f8x1`, `I.f8x2`, `I.f8x4`, `I.f8x8`, `I.f8x16`,
`I.e8m0x2`, `I.e8m0x4`, `I.e8m0x8`, `I.e8m0x16`,
`I.f4x2`, `I.f4x4`, `I.f4x8`, `I.f4x16`, `I.f4x32`

**Custom vector**: `I.vec(n, elem_type)` -> `ir.VectorType.get([n], elem_type)`

### MLIR Type Constructors (T = _mlir.extras.types)

```python
T.f16()          # ir.F16Type
T.f32()          # ir.F32Type
T.bf16()         # ir.BF16Type
T.i8()           # IntegerType.get_signless(8)
T.i32()          # IntegerType.get_signless(32)
T.i64()          # IntegerType.get_signless(64)
T.index()        # ir.IndexType
T.memref(D1, D2, ..., elem_type)             # MemRefType
T.memref(D1, D2, elem_type, memory_space=ms) # With address space
```

### FP8 Type Selection

`I.f8` automatically selects the correct FP8 variant based on GPU arch:
- gfx95*/gfx12* (MI350/RDNA4): `Float8E4M3FN` (OCP standard)
- gfx94* (MI300): `Float8E4M3FNUZ`

### Common Type Patterns

```python
DYN = ir.ShapedType.get_dynamic_size()
f32 = ir.F32Type.get()
bf16 = ir.BF16Type.get()
i32 = ir.IntegerType.get_signless(32)
i16_ty = ir.IntegerType.get_signless(16)
v8f32_ty = ir.VectorType.get([8], f32)
v8i16_ty = ir.VectorType.get([8], i16_ty)
v8bf16_ty = ir.VectorType.get([8], bf16)
v2i32_ty = ir.VectorType.get([2], i32)   # for FP8 WMMA operands
```

---

## 14. Compiler & Execution

**Import**: `import flydsl` or `from flydsl.compiler.compiler import compile`

### compile()

```python
exe = flydsl.compile(
    module,                                  # flir.MlirModule instance or ir.Module
    verify=True,                             # Enable MLIR verifier
    print_final_module=False,                # Print final MLIR
    opt_level=3,                             # LLVM optimization level (0-3)
    shared_libs=None,                        # Custom shared libraries
    use_bare_ptr_memref_call_conv=False,      # Bare pointer calling convention
    use_bare_pointers_for_host=False,
    use_bare_pointers_for_kernels=False,
)
# Returns Executor or None (if FLYDSL_COMPILE_ONLY=1)
```

### Executor

```python
# Call the compiled kernel:
exe(arg1, arg2, ..., stream_ptr)
# Arguments are ctypes pointers or numpy arrays
```

### Occupancy Control

```python
from flydsl.compiler.compiler import _apply_waves_per_eu_hint
m = MyKernel()
_apply_waves_per_eu_hint(m.module, waves_per_eu=2)  # 1-4 typical
exe = flydsl.compile(m)
```

### Pipeline Stages

```
Python DSL -> MLIR (FLIR dialect)
  -> flir-to-standard
  -> trivial-dce -> canonicalize -> cse
  -> gpu-kernel-outlining
  -> gpu.module(convert-scf-to-cf)
  -> gpu.module(convert-gpu-to-rocdl{chipset, wave64=false for RDNA})
  -> gpu.module(reconcile-unrealized-casts)
  -> rocdl-attach-target{O=2, chip=gfx1201}
  -> gpu-to-llvm
  -> reconcile-unrealized-casts
  -> gpu-module-to-binary{format=fatbin}
  -> ExecutionEngine
```

### Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `FLYDSL_TARGET_ARCH` | Override target GPU arch | auto-detect |
| `FLYDSL_COMPILE_ONLY` | "1" to compile without executor | "0" |
| `FLIR_DUMP_IR` | "1" to dump intermediate MLIR | "0" |
| `FLIR_DUMP_DIR` | Directory for IR dumps | "my_ir_dumps" |
| `FLYDSL_LLC_OPTS` | Extra LLVM LLC options | "" |

### Compilation Cache

Enabled via `FLYDSL_CACHE` env var. Uses file-based caching with hash keys.
Force rebuild with `FLYDSL_CACHE_REBUILD=1`.

---

## 15. Python Control Flow (python_control_flow)

**Import**: `from flydsl.dialects.ext.python_control_flow import range_constexpr`

### range_constexpr

Compile-time unroll: generates N copies of the body in the IR (zero loop overhead).

```python
for i in range_constexpr(8):
    # This body is duplicated 8 times in the IR
    val = vector.extract(acc, static_position=[i])
    ...
```

vs. Python `range()` in `@flir.kernel` functions, which is lowered to `scf.for`:

```python
for i in range(start, end, step):
    # This generates a single scf.for loop in the IR
    ...
```

**Rule of thumb**: Use `range_constexpr` for small fixed counts (1-16), `range()` for dynamic or large counts.

---

## 16. ROCm Helpers (rocm)

**Import**: `from flydsl.dialects.ext import rocm`

Placeholder classes for AMD GPU copy operation descriptors:

```python
rocm.CopyUniversalOp()          # Generic data movement
rocm.CopyG2LOp(vector_size=8)   # Global -> LDS
rocm.CopyL2ROp(vector_size=4)   # LDS -> Register
rocm.CopyR2GOp(vector_size=8)   # Register -> Global
rocm.CopyR2LOp(vector_size=4)   # Register -> LDS
rocm.MfmaOp(shape, a_type, b_type, c_type, arch="gfx942")  # MFMA descriptor
rocm.make_tensor(ptr, layout, element_type)
rocm.make_fragment(layout, element_type)
rocm.partition_src(tiled_copy, src_tensor, thr_idx)
rocm.partition_dst(tiled_copy, dst_tensor, thr_idx)
```

---

## Quick Reference: Common Patterns

### Pattern: Load -> WMMA -> Store (bf16)

```python
# Load preshuffle data as i16 (bf16 bitcast)
a_data = buffer_ops.buffer_load(a_rsrc, a_off, vec_width=4, dtype=i32)  # 4xi32 = 8xi16
a_i16 = vector.bitcast(v8i16_ty, a_data)
b_data = buffer_ops.buffer_load(b_rsrc, b_off, vec_width=4, dtype=i32)
b_i16 = vector.bitcast(v8i16_ty, b_data)

# WMMA (bf16 -> f32 accumulation)
acc = rocdl.wmma_f32_16x16x16_bf16(v8f32_ty, [a_i16, b_i16, arith.unwrap(prev_acc)])

# Store results
for si in range_constexpr(8):
    val = vector.extract(acc, static_position=[si])
    val_bf16 = arith.trunc_f(bf16, arith.ArithValue(val))
    buffer_ops.buffer_store(val_bf16, c_rsrc, elem_off)
```

### Pattern: Predicated Load/Store

```python
valid = arith.cmpu(row_i32, max_row_i32, "ult")
data = buffer_ops.buffer_load(rsrc, offset, mask=valid)
buffer_ops.buffer_store(result, rsrc, offset, mask=valid)
```

### Pattern: Wave-Level Reduction

```python
val = local_sum
for shift in [16, 8, 4, 2, 1]:
    shuf = gpu.ShuffleOp(
        arith.unwrap(val), arith.unwrap(arith.i32(shift)),
        arith.unwrap(arith.i32(32)), mode="xor"
    )
    val = val + arith.ArithValue(shuf.shuffleResult)
```

### Pattern: LDS Allocation and Access

```python
allocator = SmemAllocator(None, arch="gfx1201")
gen_buf = allocator.allocate_array(T.f32(), 512)
allocator.finalize()

# In init_gpu_module:
self._buf = gen_buf

# In kernel:
base = allocator.get_base()
buf_view = self._buf(base).get()
memref.store(val, buf_view, [idx])
gpu.barrier()
loaded = memref.load(buf_view, [idx])
```
