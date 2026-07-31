# Quick Start

This walkthrough gets a minimal application running. It is executable only
when [Platform Support](https://github.com/sigee-min/runD/blob/main/wiki/Platforms) lists the selected tuple as
`supported`; if no row is supported, stop rather than consuming a candidate.
The filenames below use Darwin ARM64 only as a shape example.

## 0. Verify And Install The SDK

Download the archive, checksum, and adjacent `rund-verify` from the same hosted
release. One command validates and installs the versioned prefix:

```bash
sh ./rund-verify \
  ./rund-sdk-1.0.0-darwin-arm64.tar.gz \
  ./rund-sdk-1.0.0-darwin-arm64.sha256 \
  "$PWD"
```

The destination argument is optional and defaults to the current directory.
It must already exist, and the versioned prefix must not. The verifier first
copies the archive to a private snapshot, checks its canonical SHA-256 row,
rejects unsafe paths, duplicate entries, links, and unsupported entry types,
then validates all three sealed provenance payloads. The embedded
`share/runD/release/identity.cmake` authority checks source and artifact schema,
SDK/triplet identity, and the installed public-target hash. Finally, the
verifier compares the current host, compiler, standard library, platform SDK,
and declared backend dependencies with the exact producer tuple. Only after
every check passes does it install
`$PWD/rund-sdk-1.0.0-darwin-arm64` and print the validated identity.

Any mismatch leaves the destination prefix unpublished. Do not manually
extract the archive or replace a failed check with a local dependency guess;
select an artifact whose exact tuple matches the consumer host.

## 1. Create A Minimal Project

`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(example_app LANGUAGES CXX)

find_package(runD 1.0.0 EXACT CONFIG REQUIRED)

add_executable(rund_hello main.cpp)
target_link_libraries(rund_hello PRIVATE runD::sdk)
```

`main.cpp`:

```cpp compile run source=package/tests/consumer/example/runtime.cpp
#include <rund/session.hpp>

int main() {
  const rund::Session::Result result =
      rund::run(rund::SessionConfig{.workers = 1u}, [] {});
  return result.exit_code();
}
```

## 2. Configure And Build

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$PWD/rund-sdk-1.0.0-darwin-arm64"
cmake --build build
./build/rund_hello
```

Do not point CMake at the archive, a copied `include/` directory, or a runD
source checkout. Continue with [SDK Consumption](https://github.com/sigee-min/runD/blob/main/wiki/SDK), the
[Replay first success](https://github.com/sigee-min/runD/blob/main/wiki/Replay#bind-once), or the
[Compute API](https://github.com/sigee-min/runD/blob/main/wiki/Compute).
