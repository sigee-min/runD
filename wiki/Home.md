# runD

> **Write one C++20 Flow. Run it on CPU, Metal, or Vulkan. Keep the same
> authoritative bits.**

runD is a deterministic Compute and replay SDK for systems where “close
enough” is a correctness failure. The application owns its simulation, game
state, protocol, and storage schema. runD owns the typed execution graph,
numeric law, ordering, bounded resources, evidence, and replay mechanics.

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

## The Core Idea

Declare the computation once and make backend selection explicit:

```cpp fragment
const auto execute = [&input](rund::compute::Target target) {
  return rund::compute::on(target, input)
      .map("step", [](auto value) { return value * 2 + 1; })
      .collect();
};
```

The same typed Flow produces the canonical graph for
`Target::cpu()`, `Target::metal()`, and `Target::vulkan()`. A selected backend
either executes that graph or returns a typed failure; it never silently
changes the algorithm or falls back to CPU.

See the complete, verified three-backend example in the
[repository README](https://github.com/sigee-min/runD#see-it).

## Why It Is Different

| Contract | What it gives you |
| --- | --- |
| One graph identity | Backend selection does not create a second algorithm or code path. |
| Explicit numeric law | Integer and fixed-point width, rounding, overflow, and approximation participate in execution identity. |
| Stable ordering | Scheduling, reductions, conflicts, publication, and replay observations have defined order. |
| Bounded execution | Graph size, memory, queues, compilation, iterations, retention, and failures are admitted explicitly. |
| Executable evidence | Installed contracts compare exact output bytes, hashes, graph identity, and replay evidence across available backends. |

This makes runD a fit for deterministic simulations, lockstep state, replayable
systems, reproducible Compute pipelines, and other authoritative workloads.

## Start Here

| Goal | Page |
| --- | --- |
| Install the `1.0.0` Alpha and run the first Flow | [Quick Start](https://github.com/sigee-min/runD/wiki/Start) |
| Understand bit-identical CPU/GPU execution | [Compute](https://github.com/sigee-min/runD/wiki/Compute) |
| Choose `collect`, Program, resident Job, Pipeline, or Batch | [GPU Performance](https://github.com/sigee-min/runD/wiki/Performance) |
| Record, replay, and resume canonical input | [Replay](https://github.com/sigee-min/runD/wiki/Replay) |
| Integrate the installed CMake package | [SDK Consumption](https://github.com/sigee-min/runD/wiki/SDK) |
| Browse every public entry header | [API Reference](https://github.com/sigee-min/runD/wiki/API) |
| Diagnose an installation or runtime failure | [Troubleshooting](https://github.com/sigee-min/runD/wiki/Troubleshooting) |

## Published Alpha

The public `1.0.0` Alpha ships a verified Darwin ARM64 SDK with CPU, native
Metal, and Vulkan through MoltenVK. Linux x64 is a validated source candidate,
not a supported binary release. See [Platform Support](https://github.com/sigee-min/runD/wiki/Platforms) for the
exact artifact boundary.

Applications link one CMake target, `runD::sdk`, and include focused
`<rund/*.hpp>` headers. Kernel, Accel, Node, and backend headers remain private
implementation layers.

## Determinism Boundary

Bit-identical Compute requires the documented integer or fixed-point value
domain, numeric policy, graph operations, and supported backend contract.
Arbitrary host callbacks, unconstrained floating point, wall-clock timing, and
native network arrival order do not become deterministic automatically.

Host observations become deterministic after admission into the canonical
input stream. Read [Numeric Policy](https://github.com/sigee-min/runD/blob/main/docs/architecture/numeric.md),
[Compute](https://github.com/sigee-min/runD/wiki/Compute), and [Evidence](https://github.com/sigee-min/runD/wiki/Evidence) before making an
authoritative-state claim.

This Wiki is the readable integration surface. Versioned architecture,
behavior, support, and acceptance authority remains under
[repository documentation](https://github.com/sigee-min/runD/blob/main/docs/README.md).
