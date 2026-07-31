# Math64 SIMD Contract

Math64 SIMD owns the 64-bit lane carrier primitives used by vector-first
formula families.
`core/model.hpp` owns the scalar-width vocabulary and `simd/model.hpp` owns
the public carrier and mask model consumed by the operation leaves.

Public carriers:

- `simd::I64x`: two signed 64-bit lanes.
- `simd::U64x`: two unsigned 64-bit lanes.
- `simd::Mask64x`: unsigned all-bits mask lanes.
- `simd::LaneCount`: `2`.

SIMD helpers provide lane splat, load/store, mask composition, bitwise mask
operations, selection, arithmetic, comparisons, and explicit reductions.
Unsupported compilers fail configuration instead of selecting a fallback mode.

Arithmetic law:

- `Add`, `Sub`, and `MulLow` on `U64x` are modulo-2^64 lane operations.
- `Add`, `Sub`, and `MulLow` on `I64x` use the same modulo-2^64 bit law and
  reinterpret the low 64 result bits as signed lanes. This avoids C++ signed
  overflow UB and gives deterministic wrap semantics.
- `Min`, `Max`, and comparisons use signed ordering for `I64x` and unsigned
  ordering for `U64x`.

Mask and select law:

- `Mask64x` lanes are canonical all-bits-zero or all-bits-one `u64` values.
- `MaskLaneFromBool(true)` is `0xffffffffffffffff`; false is `0`.
- `Select(mask, a, b)` returns `(mask & a) | (~mask & b)` lane-wise.

Load/store law:

- `LoadI64`, `LoadU64`, and `Store` require valid `i64*`/`u64*` storage for two
  consecutive scalar elements. The pointer must satisfy the scalar element
  alignment required by C++; 16-byte SIMD alignment is not required.
- Byte-misaligned typed pointers are outside the API precondition.

Reduction law:

- Scalar-return reductions fold lane `0`, then lane `1`, with the same
  arithmetic or bitwise lane law as the corresponding vector operation.

Public formula operations are value-transforming vector operations: arithmetic,
comparisons, selection, and mask transforms. Carrier construction, lane
construction, lane extraction, scalar-return mask folds, scalar-return
reductions, and load/store are boundary utilities covered by contract tests
rather than public formulas.

Element scalar values are storage and test literals only; they are not public
formula contracts or fallback modes.

Evidence: `math64.contract`.
