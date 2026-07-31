# runD

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

## Start Here

| Goal | Page |
| --- | --- |
| Install and run the first program | [Quick Start](https://github.com/sigee-min/runD/blob/main/wiki/Start) |
| Integrate the installed CMake package | [SDK Installation](https://github.com/sigee-min/runD/blob/main/wiki/SDK) |
| Build a typed compute graph | [Compute](https://github.com/sigee-min/runD/blob/main/wiki/Compute) |
| Record, replay, and resume state | [Replay](https://github.com/sigee-min/runD/blob/main/wiki/Replay) |
| Choose a CPU or GPU execution shape | [GPU Performance](https://github.com/sigee-min/runD/blob/main/wiki/Performance) |
| Browse the complete product surface | [API Overview](https://github.com/sigee-min/runD/blob/main/wiki/API) |
| Diagnose an integration failure | [Troubleshooting](https://github.com/sigee-min/runD/blob/main/wiki/Troubleshooting) |

## Requirements

- A runD SDK tuple listed as `supported` by
  [Platform Support](https://github.com/sigee-min/runD/blob/main/wiki/Platforms).
- CMake 3.20 or newer.
- A C++20 compiler.
- One target linked to `runD::sdk`.

The published runD 1.0.0 consumer release supports Darwin ARM64. Linux remains
a validated source candidate and is not a supported release artifact.

## Integration Boundary

Treat runD as a black-box SDK. Link `runD::sdk` `PRIVATE` when runD names stay
inside implementation files. Link the containing target `PUBLIC` only when a
documented runD type intentionally appears in that target's public
declarations.

Do not include Kernel, Accel, Node, backend, or transitive support headers.
Those layers are implementation-only.

This Wiki is the readable integration surface. Canonical architecture,
behavior, support, and performance evidence remain versioned with the source
under [repository documentation](https://github.com/sigee-min/runD/blob/main/docs/README.md).
Use [API Stability](https://github.com/sigee-min/runD/blob/main/wiki/Stability) and the
[Release Checklist](https://github.com/sigee-min/runD/blob/main/wiki/Checklist) before
changing SDK versions.
