# SDK Consumption

runD is consumed as a black-box SDK artifact. Your project should discover the
SDK through CMake and link the exported target.

## CMake Integration

```cmake
find_package(runD 1.0.0 EXACT CONFIG REQUIRED)
target_link_libraries(my_runtime PRIVATE runD::sdk) # implementation-only use
```

The imported target supplies the required C++20 compile feature, deterministic
numeric options, include closure, and libraries. Repeating those properties in
the application would create a second package contract.

The version is the exact current artifact identity. A major-only version range
is not part of the SDK contract.

Pass the prefix installed by `rund-verify` through `CMAKE_PREFIX_PATH` when configuring
your project:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$PWD/rund-sdk-1.0.0-darwin-arm64"
```

Choose that directory only after checking [Platform Support](https://github.com/sigee-min/runD/wiki/Platforms)
and completing the one-command verifier in
[Release Artifacts](https://github.com/sigee-min/runD/wiki/Artifacts). The verifier admits only the exact
compiler, standard library, platform SDK, and declared backend-dependency tuple.
Package discovery also fails when a declared backend dependency is absent; it
does not disable the backend.

## Consumption Boundary

Choose CMake visibility from your C++ boundary. Link `PRIVATE` when runD names
stay in implementation files. Link `PUBLIC` when a documented runD value or
template appears in your public headers so downstream translation units inherit
the same exact SDK contract. Both paths use the single `runD::sdk` target; the
normative examples and installed proofs are in
[SDK Consumption](https://github.com/sigee-min/runD/blob/main/package/docs/consumption.md).

Use [Public API Surface](https://github.com/sigee-min/runD/wiki/Surface) as the direct-include reference.
Support-only headers may be present to close those entries, but neither
`PRIVATE` nor `PUBLIC` promotes them to application include points.

Avoid these integration paths:

- Do not vendor runD code into your project.
- Do not add runD as a CMake subdirectory.
- Do not add runD include directories by hand.
- Do not expose source-private or undocumented runD names from public headers.
- Do not consume runD from a `thirdparty/runD` checkout.

## Boundary Example

Public header:

```cpp fragment
#pragma once

#include <cstdint>

namespace engine {

struct SessionId {
  std::uint64_t value = 0u;
};

class Runtime {
public:
  bool tick(SessionId session);
};

}  // namespace engine
```

`.cpp` file:

```cpp fragment
#include <engine/runtime.hpp>

#include <rund/session.hpp>

namespace engine {

bool Runtime::tick(SessionId) {
  const rund::Session::Result result =
      rund::run(rund::SessionConfig{.workers = 1u}, [] {});
  return static_cast<bool>(result);
}

}  // namespace engine
```

This implementation-only shape keeps runD out of the library's public ABI. A
library that intentionally publishes documented runD types instead links
`PUBLIC`; the owning consumption page proves that transitive path.

## Host Replay Storage

Long-running host I/O replay should choose storage explicitly at the embedding
layer. `Memory` is the default for small records. Use `Spill`
when replay payload bytes can outlive a short in-memory session:

```cpp fragment
rund::SessionConfig config{};
config.replay.storage.mode =
    rund::replay::StorageMode::Spill;
config.replay.storage.directory = session_temp_dir;
config.replay.storage.cached_bytes = 4u * 1024u * 1024u;
config.replay.storage.segment_bytes = 64u * 1024u * 1024u;
config.replay.storage.max_bytes =
    8ull * 1024ull * 1024ull * 1024ull;
```

The caller owns that directory's lifecycle. During replay, runD resolves
recorded payload bytes from the archive and segment files; it does not read the
native worker again if storage is missing or corrupt.

For the current public boundary, see [API Stability](https://github.com/sigee-min/runD/wiki/Stability). For
artifact verification, see [Release Checklist](https://github.com/sigee-min/runD/wiki/Checklist).
