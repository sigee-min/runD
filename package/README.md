# runD SDK Package

`/package` owns the installed SDK, exported CMake target, artifact identity,
and black-box consumer proofs. Repository include paths and subsystem targets
are private.

## Consume

```cmake
find_package(runD 1.0.1 EXACT CONFIG REQUIRED)
target_link_libraries(app PRIVATE runD::sdk)
```

Use `PUBLIC` only when a runD declaration appears in the consuming library's
public C++ boundary. Consumers do not use `add_subdirectory`, source-tree
includes, package components, or subsystem targets. The exact rules are owned
by [SDK Consumption](./docs/consumption.md).

## Surface

`runD::sdk` is the sole link target. The machine-readable direct-header and
private-header boundary is
[`docs/surface/headers.tsv`](./docs/surface/headers.tsv); installation derives
and seals its transitive physical header closure. Kernel, Accel, and Node
headers are support implementation, not direct consumer entries.

The current product vocabulary and public boundary are owned by
[API Stability](./docs/api/stability.md). Compute usage is owned by the
[Compute reference](../docs/reference/compute.md), dependent Program execution
by the [Pipeline contract](../node/docs/contracts/compute/pipeline.md), and fixed-point policy by
[Numeric Policy](../docs/architecture/numeric.md). An explicitly selected
backend executes or returns its typed failure; it never selects another
backend implicitly.

Every binary candidate carries the repository and private-dependency licenses
plus a sealed producer tuple. Exact version discovery, compiler and standard
library compatibility, native dependency identity, and supported host tuples
are owned by [Platform Support](./docs/platform/support.md).

## Docs

| Page | Owns |
| --- | --- |
| [Product Acceptance](./docs/acceptance.md) | Installed user journeys and release UX gate. |
| [SDK Surface](./docs/surface.md) | Direct headers and installed reachability. |
| [API Stability](./docs/api/stability.md) | Current public names and semantic boundary. |
| [SDK Consumption](./docs/consumption.md) | External CMake and linkage contract. |
| [Release](./docs/release.md) | Release execution, provenance, and candidate publication boundary. |
| [Artifacts](./docs/surface/artifact.md) | Installed tree, licenses, seals, and documentation snippets. |

## Verify

```sh
tools/release/run
```

This route installs a fresh prefix and configures, builds, and runs independent
external consumers through `find_package`. On Darwin ARM64, the candidate route
includes that Release gate, creates the archive, checksum, and `rund-verify`
three-file set, and self-tests the exact verifier against the extracted
archive:

```sh
tools/release/darwin
```

Hosted publication is a separate authorized operation; neither command
publishes a release.
