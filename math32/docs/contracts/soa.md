# Math32 SoA Contract

Math32 SoA APIs are the public prepared range surface. They bind existing
component arrays to vector laws without object-vector staging or hidden scalar
formula loops.

Stable surfaces:

- `soa::Add(Vec3View, Vec3View, Vec3MutView)`
- `soa::Sub(Vec3View, Vec3View, Vec3MutView)`
- `soa::Clamp(Vec3View, Vec3View, Vec3View, Vec3MutView)`
- `soa::AddInPlace(Vec3MutView, Vec3View)`

Views require equal component lengths. Component overlap within a view fails
with `StatusReason::ComponentOverlap`. Output overlap with input views fails
with `StatusReason::InputOutputOverlap`, except the explicit first-operand
in-place mutation in `AddInPlace`.

`/math32/include/math32/soa/range.hpp` is the single private byte-range
authority consumed by Geometry, SoA, NN, and Probability. For byte address
`B(s)` and byte count `N(s)`, an empty span never overlaps. Otherwise, when
`B(a) <= B(b)`, the half-open ranges overlap exactly when
`B(b) - B(a) < N(a)`; the other order is symmetric. Subtraction avoids an
overflow-prone `B + N` endpoint. Ranges are identical exactly when their byte
addresses and byte counts match, and a partial overlap is an overlap that is
not identical. Each decision is `O(1)`, performs no allocation, and copies no
payload bytes.

`StatusReason` is the single operation truth. `Status::ok()` derives success
only from `StatusReason::Ok`; a default status is `NotEvaluated`. `processed`
remains independent evidence of the number of accepted elements.

Loops advance by `simd::LaneCount`. Full chunks use `geom::Load`, vector
geometry law, and `geom::Store`. Tail chunks use fixed-width predicated lane
load/store into one vector carrier, then execute the same vector law; no scalar
formula computes tail results.

The intended engine integration shape is caller-owned structure-of-arrays
storage with separate `x`, `y`, and `z` arrays exposed as `Vec3View` or
`Vec3MutView`. This preserves unit-stride component access and avoids the
allocation and `3N` component copy cost of staging through object-vector
arrays.

Evidence: `math32.contract`.
