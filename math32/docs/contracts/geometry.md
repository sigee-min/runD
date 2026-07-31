# Math32 Geometry Contract

Math32 geometry owns `Vec2x`/`Vec3x` lane carriers, `Mask2x`/`Mask3x` masks,
SoA views, and componentwise vector laws. Prepared range-view validation and
execution surfaces are owned by the [SoA contract](./soa.md). Public math32
geometry does not expose object-value formulas, object-value range APIs, or
public 128-bit aliases or fields.

Engine semantic values live in the engine layer; runD hot-loop integration
uses `Vec2View`, `Vec2MutView`, `Vec3View`, `Vec3MutView`,
`Vec2x`, `Vec3x`, `Mask2x`, and `Mask3x`.

View validation rejects mismatched component lengths and overlapping
components. `Load`/`Store` bridge validated SoA spans to `Vec*x` carriers at an
explicit lane base; callers must check `CanLoad` for reads and `CanStore` for
writes, or prove prepared padded capacity before the hot path.

`ViewStatusReason` is the single validation truth. `ViewStatus::ok()` and the
boolean conversion derive success only from `ViewStatusReason::Ok`; a default
status is `NotEvaluated`. The status carries one reason and the validated size.

The canonical production law is the SIMD lane law for this product width.
Element scalar values are storage and test literals only; they are not public
formula contracts or fallback modes.

Evidence: `math32.contract`.
