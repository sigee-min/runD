# runD

<p align="center">
  <strong>Deterministic execution for simulations, games, and replayable systems.</strong>
</p>

<p align="center">
  <a href="https://github.com/sigee-min/runD/releases"><img alt="Release" src="https://img.shields.io/github/v/release/sigee-min/runD?include_prereleases"></a>
  <img alt="Status" src="https://img.shields.io/badge/status-alpha-f0ad4e">
  <img alt="C++20" src="https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus">
  <img alt="Backends" src="https://img.shields.io/badge/backends-CPU%20%7C%20Metal%20%7C%20Vulkan-6C5CE7">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

runD is a C++20 SDK for deterministic Compute, replay, bounded runtime work,
networking, telemetry, and optional cluster placement. Applications keep
ownership of physics, game state, protocols, and storage schemas; runD owns
execution order, numeric policy, evidence, and replay mechanics.

```text
canonical input → runD Pipeline → deterministic state
                       ├─ live
                       ├─ record / replay
                       └─ CPU / Metal / Vulkan
```

[Quick start](#quick-start) ·
[API overview](wiki/API.md) ·
[Compute](wiki/Compute.md) ·
[Replay](wiki/Replay.md) ·
[Documentation](docs/README.md) ·
[Issues](https://github.com/sigee-min/runD/issues) ·
[Releases](https://github.com/sigee-min/runD/releases)

## Alpha status

`1.0.0` is the first public alpha. The versioned SDK surface, package
inventory, and deterministic execution contracts are tested, but compatibility
may still change before a stable release.

| Platform | Release status | Backends |
| --- | --- | --- |
| Darwin ARM64 | Supported binary SDK | CPU, Metal, Vulkan through MoltenVK |
| Linux x64 | Validated source candidate | CPU, Vulkan when available |
| Windows x64 | Not supported | — |

The Darwin archive is intentionally strict: `rund-verify` admits only a host
whose compiler, standard library, Apple SDK, architecture, and native backend
dependencies match the sealed producer tuple. Large accelerator workloads
must also be admitted against real device and driver memory headroom; the
alpha does not claim production readiness for every graph that can be
compiled or prepared.

The complete platform contract lives in
[Platform Support](package/docs/platform/support.md). Please report unexpected
behavior through [GitHub Issues](https://github.com/sigee-min/runD/issues).

## Why runD?

- **Bit-stable execution** — fixed ordering, explicit numeric policy, stable
  graph identity, and backend parity contracts.
- **Replay without a second graph** — Live, Record, Replay, and Scenario share
  one canonical input boundary and the same downstream Pipeline.
- **Typed Compute** — build a Flow once, select CPU, Metal, or Vulkan
  explicitly, and reuse prepared resident execution.
- **Bounded by design** — memory, compilation, queues, retention, and failure
  behavior are admitted before execution.
- **Actionable telemetry** — allocation, copy, queue pressure, scan, and
  critical-path findings include a cause and a concrete action.
- **Domain neutral** — runD owns execution mechanics and evidence, never
  application-specific world, body, command, or protocol types.

## Quick start

Download these three files from the
[1.0.0 Alpha release](https://github.com/sigee-min/runD/releases/tag/1.0.0):

```text
rund-sdk-1.0.0-darwin-arm64.tar.gz
rund-sdk-1.0.0-darwin-arm64.sha256
rund-verify
```

Verify the sealed identity and install the SDK into the current directory:

```sh
chmod +x ./rund-verify
sh ./rund-verify \
  ./rund-sdk-1.0.0-darwin-arm64.tar.gz \
  ./rund-sdk-1.0.0-darwin-arm64.sha256 \
  "$PWD"
```

Create a small CMake consumer:

```cmake
cmake_minimum_required(VERSION 3.20)
project(example LANGUAGES CXX)

find_package(runD 1.0.0 EXACT CONFIG REQUIRED)

add_executable(example main.cpp)
target_link_libraries(example PRIVATE runD::sdk)
```

```cpp compile run
#include <rund/compute.hpp>

#include <array>
#include <cstdint>
#include <vector>

int main() {
  std::array<std::int32_t, 4> input{1, 2, 3, 4};

  auto output = rund::compute::on(rund::compute::Target::cpu(), input)
                    .map("twice", [](auto value) { return value * 2; })
                    .collect();

  if (!output) {
    return output.exit_code();
  }
  return *output == std::vector<std::int32_t>{2, 4, 6, 8} ? 0 : 2;
}
```

Build it against the verified prefix:

```sh
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/path/to/rund-sdk-1.0.0-darwin-arm64
cmake --build build
./build/example
```

See the versioned [Quick Start](wiki/Start.md) and
[SDK Consumption](package/docs/consumption.md) contracts for the complete
artifact and CMake rules.

## Build from source

Source development requires CMake 3.20 or newer, Ninja, a C++20 compiler, and
the platform dependencies for the backends being enabled. Vulkan discovery
can be disabled for a CPU-only development build.

```sh
cmake -S . -B .cache/build/local -G Ninja \
  -DRUND_ENABLE_VULKAN=OFF
cmake --build .cache/build/local
ctest --test-dir .cache/build/local --output-on-failure
```

Repository verification uses the checked-in operators:

```sh
tools/test/run --list
tools/test/run <case>
tools/check/run
tools/release/run
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) and the
[verification contract](docs/architecture/verification.md) before changing
behavior.

## Core APIs

| API | Purpose | Start here |
| --- | --- | --- |
| Session | Reusable lifecycle, tasks, and bounded resources | [Runtime contract](node/docs/contracts/runtime.md) |
| Compute | Typed Flow, Program, Job, and Pipeline execution | [Compute guide](wiki/Compute.md) |
| Replay | Canonical input, checkpoints, scenarios, and evidence | [Replay guide](wiki/Replay.md) |
| Network | Move-only sockets and typed asynchronous I/O | [Network contract](node/docs/contracts/net.md) |
| Telemetry | Stable counters and bounded actionable findings | [Telemetry guide](wiki/Telemetry.md) |
| Performance | GPU sizing, resident execution, fusion, and batching | [GPU guide](wiki/Performance.md) |
| Fixed | Widened fixed-point expressions and explicit quantization | [Numeric policy](docs/architecture/numeric.md) |

Use the focused `<rund/*.hpp>` header for the API you need. `<rund/rund.hpp>`
is reserved for translation units that intentionally compose the complete SDK.
Applications link only `runD::sdk`; Kernel, Accel, and Node are private layers.

## Architecture

| Layer | Responsibility |
| --- | --- |
| [`math32/`](math32/docs/README.md), [`math64/`](math64/docs/README.md) | Fixed-width arithmetic and bit-level laws |
| [`kernel/`](kernel/docs/README.md) | Scheduling, workspaces, reductions, and execution order |
| [`accel/`](accel/docs/README.md) | Backend-neutral accelerator values |
| [`node/`](node/docs/README.md) | Runtime, Compute bridge, networking, replay, and telemetry |
| [`cluster/`](cluster/docs/README.md) | Optional placement and retry identity |
| [`package/`](package/README.md) | Installed SDK and artifact contract |

The repository-wide architecture and engineering source of truth starts at
[`docs/README.md`](docs/README.md).

## Contributing and security

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md) and
keep behavior, documentation, and contract evidence in the same change.
Security reports should follow [SECURITY.md](SECURITY.md), not a public issue.

## License

runD is released under the [MIT License](LICENSE). The bundled xxHash header
retains its own notice at
[`node/src/host/vendor/xxhash/LICENSE`](node/src/host/vendor/xxhash/LICENSE).
