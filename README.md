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
  <a href="https://sigee-min.github.io/runD/">Website</a> ·
  <a href="https://sigee-min.github.io/runD/docs/start/">Quick start</a> ·
  <a href="https://sigee-min.github.io/runD/docs/">Docs</a> ·
  <a href="https://sigee-min.github.io/runD/docs/api/">API &amp; errors</a> ·
  <a href="https://github.com/sigee-min/runD/releases">Download</a>
</p>

## See it

The algorithm below is declared once. Only the explicit target changes.

```cpp compile run source=package/tests/consumer/example/parity.cpp
#include <rund/compute.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

int main() {
  const std::array<std::int32_t, 4> input{1, 2, 3, 4};

  const auto execute = [&input](const rund::compute::Device &device) {
    auto flow = rund::compute::on(device).map<std::int32_t>(
        "step", input.size(), [](auto value) { return value * 2 + 1; });
    auto program = std::move(flow).compile();
    if (!program) {
      return rund::compute::Result<std::vector<std::int32_t>>::fail(
          program.reason());
    }
    return program->run(std::span<const std::int32_t>{input});
  };

  const auto report = [](const std::string_view target, const auto &result) {
    const std::string_view message = result.error();
    std::fprintf(stderr, "%.*s failed (code=%u): %.*s\n",
                 static_cast<int>(target.size()), target.data(),
                 static_cast<unsigned>(result.code()),
                 static_cast<int>(message.size()), message.data());
    return result.exit_code();
  };
  auto cpu_device = rund::compute::open(rund::compute::Target::cpu());
  if (!cpu_device) {
    return report("cpu", cpu_device);
  }
  const auto cpu = execute(*cpu_device);
  if (!cpu) {
    return report("cpu", cpu);
  }

  const auto same_bits = [](const auto &left, const auto &right) {
    return left.size() == right.size() &&
           std::memcmp(left.data(), right.data(),
                       left.size() * sizeof(std::int32_t)) == 0;
  };
  const std::vector<std::int32_t> expected{3, 5, 7, 9};
  if (*cpu != expected) {
    std::fputs("cpu output mismatch\n", stderr);
    return 2;
  }

  std::size_t available_backend_count = 1u;
#if defined(__APPLE__) && (defined(__arm64__) || defined(__aarch64__))
  constexpr bool require_accelerators = true;
#else
  constexpr bool require_accelerators = false;
#endif
  const auto compare = [&](const std::string_view name,
                           const rund::compute::Target target) {
    auto device = rund::compute::open(target);
    if (!device) {
      if (!require_accelerators &&
          device.reason() == rund::compute::Reason::AdapterUnavailable) {
        const std::string_view message = device.error();
        std::fprintf(stderr, "%.*s unavailable (code=%u): %.*s\n",
                     static_cast<int>(name.size()), name.data(),
                     static_cast<unsigned>(device.code()),
                     static_cast<int>(message.size()), message.data());
        return 0;
      }
      return report(name, device);
    }
    const auto result = execute(*device);
    if (!result) {
      return report(name, result);
    }
    if (!same_bits(*cpu, *result)) {
      std::fprintf(stderr, "%.*s output mismatch\n",
                   static_cast<int>(name.size()), name.data());
      return 2;
    }
    ++available_backend_count;
    return 0;
  };
  if (const int metal = compare("metal", rund::compute::Target::metal());
      metal != 0) {
    return metal;
  }
  if (const int vulkan = compare("vulkan", rund::compute::Target::vulkan());
      vulkan != 0) {
    return vulkan;
  }

  if (available_backend_count == 3u) {
    std::printf("same bytes: cpu = metal = vulkan [%d, %d, %d, %d]\n",
                (*cpu)[0], (*cpu)[1], (*cpu)[2], (*cpu)[3]);
  } else {
    std::printf("verified [%d, %d, %d, %d] on %zu native backend(s)\n",
                (*cpu)[0], (*cpu)[1], (*cpu)[2], (*cpu)[3],
                available_backend_count);
  }
  return 0;
}
```

There is no backend branch inside the computation, no backend-specific graph,
and no silent CPU fallback. Device opening is the only capability boundary. A
non-Apple candidate may report the exact `AdapterUnavailable` reason there;
once a device opens, every compile, execution, and byte comparison must pass.
The supported Darwin ARM64 tuple requires CPU, Metal, and Vulkan and preserves
the strict three-way output above. A headless non-Apple host remains an
executable example without pretending that an unavailable accelerator ran.

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

Read the public [Determinism](https://sigee-min.github.io/runD/docs/determinism/),
[Numerics](https://sigee-min.github.io/runD/docs/numerics/), and
[Compute](https://sigee-min.github.io/runD/docs/compute/) guides behind that
claim.

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

The versioned
[Quick Start](https://sigee-min.github.io/runD/docs/start/) is the public
integration authority. The compact path below previews its verified Darwin
ARM64 installation. Download these three adjacent files from
[Releases](https://github.com/sigee-min/runD/releases/tag/1.0.4):

Before continuing, check the exact native dependency tuple in the Quick Start.
The 1.0.4 release does not bundle MoltenVK or publish a versioned Homebrew tap,
so current unversioned formulas are not a substitute when their versions have
moved past the sealed artifact identity.

```text
rund-sdk-1.0.4-darwin-arm64.tar.gz
rund-sdk-1.0.4-darwin-arm64.sha256
rund-verify
```

Verify the sealed producer tuple and install the SDK:

```sh
chmod +x ./rund-verify
sh ./rund-verify \
  ./rund-sdk-1.0.4-darwin-arm64.tar.gz \
  ./rund-sdk-1.0.4-darwin-arm64.sha256 \
  "$PWD"
```

Link the single public target:

```cmake
cmake_minimum_required(VERSION 3.20)
project(example LANGUAGES CXX)

find_package(runD 1.0.4 EXACT CONFIG REQUIRED)

add_executable(example main.cpp)
target_link_libraries(example PRIVATE runD::sdk)
```

```sh
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$PWD/rund-sdk-1.0.4-darwin-arm64"
cmake --build build
./build/example
```

Continue with [Troubleshooting](https://sigee-min.github.io/runD/docs/troubleshooting/)
if setup fails, or the public
[Compute guide](https://sigee-min.github.io/runD/docs/compute/) after the first
successful Flow.

## Determinism boundary

runD does not claim that arbitrary host code, unconstrained floating-point
expressions, wall-clock timing, or native event arrival order become
deterministic automatically. Bit-identical authority requires an admitted
numeric contract and the documented deterministic graph operations.

Host observations become deterministic after admission into the canonical
input stream. Floating-point state must use the strict admitted law; diagnostic
or presentation floating point cannot feed authoritative state.

## Alpha platform status

`1.0.4` is the current public alpha; `1.0.0` was the first. The SDK surface and
deterministic contracts are tested, but compatibility may change before a
stable release.

| Platform | Status | Backends |
| --- | --- | --- |
| Darwin ARM64 | Supported binary SDK | CPU, Metal, Vulkan through MoltenVK |
| Linux x64 | Validated source candidate | No release-backend matrix claimed |
| Windows x64 | Not supported | — |

The Darwin verifier admits only a host whose compiler, standard library,
Apple SDK, architecture, and native backend dependencies match the sealed
producer tuple. Large accelerator workloads must also be tested against actual
device and driver memory headroom; compilation or preparation alone is not a
production-capacity guarantee.

See public [Platform Support](https://sigee-min.github.io/runD/docs/platforms/)
for the exact integration boundary.

## Core APIs

| API | Purpose | Start here |
| --- | --- | --- |
| Compute | Typed Flow, Program, Job, Batch, and Pipeline execution | [Compute guide](https://sigee-min.github.io/runD/docs/compute/) |
| Replay | Canonical input, checkpoints, scenarios, and evidence | [Replay guide](https://sigee-min.github.io/runD/docs/replay/) |
| Session | Reusable lifecycle, tasks, and bounded resources | [Runtime guide](https://sigee-min.github.io/runD/docs/runtime/) |
| Fixed | Widened fixed-point expressions and explicit quantization | [Numerics guide](https://sigee-min.github.io/runD/docs/numerics/) |
| Network | Move-only sockets and typed asynchronous I/O | [Runtime guide](https://sigee-min.github.io/runD/docs/runtime/) |
| Telemetry | Stable counters and bounded actionable findings | [Runtime guide](https://sigee-min.github.io/runD/docs/runtime/) |

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
