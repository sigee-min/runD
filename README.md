<h1 align="center">runD</h1>

<p align="center">
  <strong>One computation. CPU, Metal, Vulkan. The same bits.</strong>
</p>

<p align="center">
  Write the deterministic path once, choose the backend at runtime,<br>
  and keep authoritative results bit-for-bit identical across supported targets.
</p>

<p align="center">
  <a href="https://github.com/sigee-min/runD/releases"><img alt="Release" src="https://img.shields.io/github/v/release/sigee-min/runD?include_prereleases"></a>
  <img alt="Status" src="https://img.shields.io/badge/status-alpha-f0ad4e">
  <img alt="C++20" src="https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus">
  <img alt="Backends" src="https://img.shields.io/badge/CPU%20%7C%20Metal%20%7C%20Vulkan-same%20graph-6C5CE7">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

```text
                         one typed Flow
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
                CPU          Metal         Vulkan
                 │             │             │
                 └─────────────┼─────────────┘
                               │
                    identical state bytes
```

<p align="center">
  <a href="#see-it">See it</a> ·
  <a href="#why-the-bits-match">Why it works</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="wiki/API.md">API</a> ·
  <a href="docs/README.md">Docs</a> ·
  <a href="https://github.com/sigee-min/runD/releases">Download</a>
</p>

## See it

The algorithm below is declared once. Only the explicit target changes.

```cpp compile run
#include <rund/compute.hpp>

#include <array>
#include <cstdint>
#include <cstring>

int main() {
  const std::array<std::int32_t, 4> input{1, 2, 3, 4};

  const auto execute = [&input](rund::compute::Target target) {
    return rund::compute::on(target, input)
        .map("step", [](auto value) { return value * 2 + 1; })
        .collect();
  };

  const auto cpu = execute(rund::compute::Target::cpu());
  const auto metal = execute(rund::compute::Target::metal());
  const auto vulkan = execute(rund::compute::Target::vulkan());

  if (!cpu) {
    return cpu.exit_code();
  }

  const auto same_bits = [](const auto& a, const auto& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(),
                       a.size() * sizeof(a[0])) == 0;
  };

  const auto matches_or_unavailable = [&cpu, &same_bits](const auto& result) {
    return result ? same_bits(*cpu, *result)
                  : result.code() == rund::compute::Code::Unavailable;
  };

  return matches_or_unavailable(metal) && matches_or_unavailable(vulkan) ? 0
                                                                         : 2;
}
```

There is no backend branch inside the computation, no backend-specific graph,
and no silent CPU fallback. A selected backend either executes that graph or
returns a typed failure. The portable executable accepts an absent accelerator
only through the typed `Unavailable` code; every available backend must return
the same bytes as CPU.

## Why the bits match

runD treats determinism as an execution contract rather than a compiler flag.

| Contract | What runD fixes |
| --- | --- |
| **One graph identity** | The same typed Flow produces one canonical graph and fingerprint for every backend. |
| **Explicit numeric law** | Integer, fixed-point, rounding, overflow, and strict floating-point policy are part of admission—not ambient compiler behavior. |
| **Stable execution order** | Scheduling, reductions, conflicts, publication, and replay observations have defined ordering. |
| **No hidden fallback** | CPU, Metal, and Vulkan are selected explicitly; failure cannot silently change the execution path. |
| **Proof, not hope** | Installed contracts compare exact output bytes, state hashes, graph identity, and replay evidence across available backends. |

This is what makes runD useful for simulations, deterministic game state,
lockstep systems, reproducible compute pipelines, and any workload where
“close enough” is a bug.

```text
same input
  + same numeric contract
  + same graph identity
  + same ordered execution
  = same authoritative bits
```

Read the exact [numeric policy](docs/architecture/numeric.md),
[Compute contract](docs/reference/compute.md), and
[release acceptance](package/docs/acceptance.md) behind that claim.

## What you get

- **Write once, run explicitly** — one C++20 Flow for CPU, Metal, and Vulkan.
- **Resident GPU execution** — compile and prepare once, then reuse Programs,
  Jobs, Batches, and dependent Pipelines.
- **Replay without a second code path** — Live, Record, Replay, and Scenario
  share one canonical input boundary.
- **Bounded by construction** — memory, graph size, compilation, queues,
  iterations, retention, and failure behavior are admitted before execution.
- **Typed failures** — unsupported hardware, exhausted capacity, device loss,
  and invalid state stay machine-readable.
- **Actionable evidence** — hashes, counters, memory plans, profiles, and
  telemetry show what executed and why.

## Quick start

The `1.0.1` Alpha ships a verified Darwin ARM64 SDK. Download these three
adjacent files from [Releases](https://github.com/sigee-min/runD/releases/tag/1.0.1):

```text
rund-sdk-1.0.1-darwin-arm64.tar.gz
rund-sdk-1.0.1-darwin-arm64.sha256
rund-verify
```

Verify the sealed producer tuple and install the SDK:

```sh
chmod +x ./rund-verify
sh ./rund-verify \
  ./rund-sdk-1.0.1-darwin-arm64.tar.gz \
  ./rund-sdk-1.0.1-darwin-arm64.sha256 \
  "$PWD"
```

Link the single public target:

```cmake
cmake_minimum_required(VERSION 3.20)
project(example LANGUAGES CXX)

find_package(runD 1.0.1 EXACT CONFIG REQUIRED)

add_executable(example main.cpp)
target_link_libraries(example PRIVATE runD::sdk)
```

```sh
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/path/to/rund-sdk-1.0.1-darwin-arm64
cmake --build build
./build/example
```

Continue with the versioned [Quick Start](wiki/Start.md),
[Compute guide](wiki/Compute.md), or
[SDK Consumption](package/docs/consumption.md).

## Determinism boundary

runD does not claim that arbitrary host code, unconstrained floating-point
expressions, wall-clock timing, or native event arrival order become
deterministic automatically. Bit-identical authority requires an admitted
numeric contract and the documented deterministic graph operations.

Host observations become deterministic after admission into the canonical
input stream. Floating-point state must use the strict admitted law; diagnostic
or presentation floating point cannot feed authoritative state.

## Alpha platform status

`1.0.1` is the current public alpha; `1.0.0` was the first. The SDK surface and
deterministic contracts are tested, but compatibility may change before a stable release.

| Platform | Status | Backends |
| --- | --- | --- |
| Darwin ARM64 | Supported binary SDK | CPU, Metal, Vulkan through MoltenVK |
| Linux x64 | Validated source candidate | CPU, Vulkan when available |
| Windows x64 | Not supported | — |

The Darwin verifier admits only a host whose compiler, standard library,
Apple SDK, architecture, and native backend dependencies match the sealed
producer tuple. Large accelerator workloads must also be tested against actual
device and driver memory headroom; compilation or preparation alone is not a
production-capacity guarantee.

See [Platform Support](package/docs/platform/support.md) for the exact boundary.

## Core APIs

| API | Purpose | Start here |
| --- | --- | --- |
| Compute | Typed Flow, Program, Job, Batch, and Pipeline execution | [Compute guide](wiki/Compute.md) |
| Replay | Canonical input, checkpoints, scenarios, and evidence | [Replay guide](wiki/Replay.md) |
| Session | Reusable lifecycle, tasks, and bounded resources | [Runtime contract](node/docs/contracts/runtime.md) |
| Fixed | Widened fixed-point expressions and explicit quantization | [Numeric policy](docs/architecture/numeric.md) |
| Network | Move-only sockets and typed asynchronous I/O | [Network contract](node/docs/contracts/net.md) |
| Telemetry | Stable counters and bounded actionable findings | [Telemetry guide](wiki/Telemetry.md) |

Use the focused `<rund/*.hpp>` header for the API you need. Applications link
only `runD::sdk`; Kernel, Accel, and Node remain private implementation layers.

## Build from source

Source development requires CMake 3.20 or newer, Ninja, a C++20 compiler, and
the dependencies for each enabled backend. Vulkan discovery can be disabled
for a CPU-only development build.

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

## Architecture

| Layer | Responsibility |
| --- | --- |
| [`math32/`](math32/docs/README.md), [`math64/`](math64/docs/README.md) | Fixed-width arithmetic and bit-level laws |
| [`kernel/`](kernel/docs/README.md) | Scheduling, workspaces, reductions, and execution order |
| [`accel/`](accel/docs/README.md) | Backend-neutral accelerator values |
| [`node/`](node/docs/README.md) | Runtime, Compute bridge, networking, replay, and telemetry |
| [`cluster/`](cluster/docs/README.md) | Optional placement and retry identity |
| [`package/`](package/README.md) | Installed SDK and artifact contract |

The repository-wide source of truth starts at
[`docs/README.md`](docs/README.md).

## Contributing and security

Contributions are welcome. Keep behavior, documentation, and executable
evidence in the same change; start with [CONTRIBUTING.md](CONTRIBUTING.md).
Report security issues through the private process in
[SECURITY.md](SECURITY.md), not a public issue.

## License

runD is released under the [MIT License](LICENSE). The bundled xxHash header
retains its own [license notice](node/src/host/vendor/xxhash/LICENSE).
