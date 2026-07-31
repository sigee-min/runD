# Math32 SIMD Contract

Math32 SIMD owns the 32-bit lane carrier primitives used by vector-first
formula families.
`core/model.hpp` owns the scalar-width vocabulary and `simd/model.hpp` owns
the public carrier and mask model consumed by the operation leaves.

Public carriers:

- `simd::I32x`: four signed 32-bit lanes.
- `simd::U32x`: four unsigned 32-bit lanes.
- `simd::Mask32x`: unsigned all-bits mask lanes.
- `simd::LaneCount`: `4`.

SIMD helpers provide lane splat, load/store, mask composition, bitwise mask
operations, selection, arithmetic, comparisons, and explicit reductions.
Unsupported compilers fail configuration instead of selecting a fallback mode.

Public formula operations are value-transforming vector operations: arithmetic,
comparisons, selection, and mask transforms. Carrier construction, lane
construction, lane extraction, scalar-return mask folds, scalar-return
reductions, and load/store are boundary utilities covered by contract tests
rather than public formulas.

Element scalar values are storage and test literals only; they are not public
formula contracts or fallback modes.

Evidence: `math32.contract`.
