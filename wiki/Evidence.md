# Evidence API

Entry header: `<rund/evidence.hpp>`

Namespace: `rund::evidence`

[Back to API Reference](https://github.com/sigee-min/runD/wiki/API)

## When To Use

Use this header when a deterministic run needs a canonical numeric policy,
stable identity, or portable evidence text. Ordinary callers should start from
a preset such as `strict_f32()` or `fixed<I, F>()`; raw `Contract` construction
is available when every policy axis must be explicit.

## Key Types

| Type | Purpose |
| --- | --- |
| `Contract` | Domain, arithmetic, authority, determinism, and fixed-point policy. |
| `Id` | Stable nonzero identity derived from a valid contract. |
| `Numeric` | Validated contract plus one typed build/decode outcome. |

## Key Functions

| Function | Purpose |
| --- | --- |
| `valid(contract)` | Check every numeric policy invariant. |
| `identify(contract)` | Derive the canonical identity, or zero for an invalid contract. |
| `make(contract)` | Validate a contract and build `Numeric`. |
| `encode(numeric)` | Encode canonical evidence text. |
| `decode(text)` | Decode and validate canonical evidence text. |

The common presets are `i32()`, `i64()`, `fixed<I, F>()`, `strict_f32()`,
`strict_f64()`, `diagnostic_f32()`, `diagnostic_f64()`,
`presentation_f32()`, and `presentation_f64()`.

## Result Rules

`Numeric::Code` is the sole outcome authority. Its values are `Ok`,
`NotBuilt`, `BadHeader`, `DuplicateField`, `MissingField`, `BadValue`, and
`HashInvalid`. Truth conversion, `ok()`, `error()`, and `exit_code()` derive
from that code. Identity, contract, and hash observers are meaningful only for
`Ok`. `strict_float()` reports whether the admitted arithmetic law requires
strict floating-point execution.

## Example

```cpp compile run
#include <rund/evidence.hpp>

int main() {
  const auto numeric = rund::evidence::make(rund::evidence::strict_f32());
  if (!numeric) {
    return numeric.exit_code();
  }
  const auto decoded = rund::evidence::decode(rund::evidence::encode(numeric));
  if (!decoded) {
    return decoded.exit_code();
  }
  return decoded.id() == numeric.id() ? 0 : 2;
}
```
