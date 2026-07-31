# Troubleshooting

## An Operation Returns False

Every documented Session and Replay terminal value has one typed `code()`
authority. Use that value for control flow and aggregation, present the
derived `error()` text to a person, and return `exit_code()` at a process
boundary. Do not parse or compare error text: it is a diagnostic projection,
not a second status protocol. A telemetry event reports the same typed outcome
and does not replace the operation result.

| Action | Use |
| --- | --- |
| Retry, classify, count, or switch | `code()` |
| Log or display the stable explanation | `error()` |
| Return from `main` or a child process | `exit_code()` |

Official examples reserve process exit `2` for their own value or invariant
check after every product result succeeded. A product failure is never remapped
to that code; it returns the failing value's `exit_code()` directly.

When reporting a failure, include the operation, typed code, derived text, SDK
version, artifact identity, and selected backend. This is enough to distinguish
an input, capacity, lifecycle, storage, or backend rejection without exposing
source-private state.

## CMake Cannot Find runD

Symptom:

```text
Could not find a package configuration file provided by "runD"
```

Fix:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$PWD/rund-sdk-1.0.0-darwin-arm64"
```

Pass the prefix installed by `rund-verify`, not the archive file.

Confirm that verified prefix contains:

```text
rund-sdk-1.0.0-darwin-arm64/lib/cmake/runD/runDConfig.cmake
```

If `runDConfig.cmake` is missing, discard that directory and run the coherent
archive/checksum/`rund-verify` release set into a fresh destination. Do not
point CMake at an application build directory or a copied `include/`
directory.

## Header Include Fails

Symptom:

```text
fatal error: rund/session.hpp: No such file or directory
```

Fix:

```cmake
find_package(runD 1.0.0 EXACT CONFIG REQUIRED)
target_link_libraries(my_runtime PRIVATE runD::sdk)
```

Let the imported CMake target provide include paths. Do not hard-code include
directories.

## Link Or Symbol Mismatch

Symptom:

```text
undefined reference
```

or:

```text
symbol not found
```

Fix:

- Confirm that your target links `runD::sdk`.
- Confirm that your compiler, standard library, build type, external backend
  dependency versions, and SDK version match the artifact you downloaded.
- Clean your build directory after switching SDK artifacts.

## Tuple Mismatch

Symptom:

```text
runD was found, but the executable fails during link or startup
```

Fix:

- Run the adjacent `rund-verify` against a fresh destination. It compares the
  platform triplet, compiler family and version, standard library, platform
  SDK, and external backend dependencies before publishing a prefix.
- Download the artifact built for the exact tuple reported by the current
  host.
- Reconfigure from a clean build directory after changing artifacts.

## SDK Verification Failure

Symptom:

```text
runD SDK ... mismatch
```

Fix:

- Remove the archive, `.sha256`, and `rund-verify`, then download one coherent
  three-file set from the same release.
- Run `sh ./rund-verify <archive> <checksum> [destination]`; do not manually
  unpack or bypass its archive, seal, schema, public-target, or host-tuple
  checks.
- If the files are intact but the tuple differs, select the supported artifact
  for the current compiler, standard library, platform SDK, and backend
  dependencies. A failed verifier publishes no SDK prefix.

## Public runD Type With A Private Link

Symptom:

```cpp compile
#include <rund/session.hpp>

struct AppRuntimeSettings {
  rund::SessionConfig config;
};
```

Fix:

Choose the boundary you intend:

- To keep runD out of your ABI, move its headers and names into `.cpp` files
  and expose application-owned types.
- To expose documented runD values or templates intentionally, link
  `runD::sdk` as `PUBLIC` so every downstream target inherits the exact SDK
  requirements.

Do not publish a runD type while linking the containing library `PRIVATE`; see
[SDK Consumption](https://github.com/sigee-min/runD/blob/main/package/docs/consumption.md) for both proven shapes.

## Wrong Platform Artifact

Symptom:

```text
file format not recognized
```

or a loader error when the executable starts.

Fix:

Do not substitute an archive merely because the source can configure on that
platform. The canonical platform policy must list the tuple as `supported`,
and the SDK version, compiler, standard library, build type, public compile
definitions, and external backend dependency versions must all match the
producing tuple. Use the embedded sealed artifact identity as that tuple's
binary identity authority.

See [Platform Support](https://github.com/sigee-min/runD/wiki/Platforms), [Release Artifacts](https://github.com/sigee-min/runD/wiki/Artifacts),
and [Release Checklist](https://github.com/sigee-min/runD/wiki/Checklist) before retrying with a different
archive.
