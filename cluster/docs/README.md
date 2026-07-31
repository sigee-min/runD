# Cluster Docs

`/cluster` is the optional distributed policy plane. It exposes
`rund::cluster`, consumes only `rund::evidence::Id` from the numeric evidence
contract, and must not consume `/kernel` directly. No `rund::node` type is part
of the Cluster surface.

## Authority

1. [`/docs/architecture/topology.md`](../../docs/architecture/topology.md)
2. This page for the cluster boundary.
3. [Contracts](./contracts/README.md).
4. Public headers under `/cluster/include/cluster`.
5. Implementation under `/cluster/src`.
6. Contract tests under `/cluster/tests/contract`.
7. Numeric evidence policy under
   [`/docs/architecture/numeric.md`](../../docs/architecture/numeric.md).

## Boundary

Cluster owns shard-to-node placement, placement epoch identity, and retry
identity policy.

Cluster does not own node admission internals, node resource envelopes, node
topology truth, kernel packet identity, kernel partition placement, kernel fold
order, or kernel telemetry truth.

`<rund/evidence/numeric/contract.hpp>` is the one declaration authority for the
numeric id carried by `RunKey`, so the build-tree Cluster target publishes the
evidence include root. It does not publish a Node link dependency: a declaration
dependency is not a symbol dependency. Cluster owns every distributed identity
itself. Repository contracts compile the three Cluster implementation units
once as one OBJECT owner and link only the test assertion owner. Placement,
retry, and run identity call no Node, Math, Runtime, Compute, or accelerator
symbol.

## Contents

| Page | Owns |
| --- | --- |
| [Contracts](./contracts/README.md) | Stable cluster behavior. |
| [Run identity](./contracts/run.md) | Deterministic distributed run identity. |
| [Placement](./contracts/placement.md) | Shard-to-node placement. |
| [Retry](./contracts/retry.md) | Retry identity over Cluster run keys. |

## Rules

- Cluster does not admit work into the kernel.
- Retry may preserve run identity only when the Cluster run key is unchanged or
  proven equivalent.
- Dependency direction is `rund::cluster -> rund::evidence`; Cluster does not
  enter Runtime or Kernel execution paths.

## Verification

The commands below are an owner-local subsystem development loop only; they do
not provide Release, source-stability, installed-package, or artifact evidence.
Use `tools/release/run` for those repository-wide claims.

- `cmake -S . -B .cache/build/cluster-contract-route -DRUND_ENABLE_CLUSTER_CONTRACT_TESTS=ON`
- `cmake --build .cache/build/cluster-contract-route --target test_cluster`
- `ctest --test-dir .cache/build/cluster-contract-route -R '^cluster[.]contract$' --output-on-failure --no-tests=error`

`test_cluster` is created only by the cluster contract route. The default
development configure does not expose the target or register its CTest row.

## SDK Artifact

Release packaging for this subsystem is owned by
[`/package`](../../package/README.md). This subsystem owns behavior and tests;
`/package` owns artifact layout, CMake package export, and external
consumption policy.
