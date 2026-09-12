# Virtual Performance

This directory owns current-source VirtualPipeline measurement procedure and
results. It is separate from the installed Release baseline.

| Page | Owns |
| --- | --- |
| [Method](./method.md) | Workload, sampling, counter admission, crossover classification, and blocked-target rules. |
| [Window](./window.md) | Direct pointwise Q2--Q4 paired CPU/Metal measurement and its public-receipt boundary. |
| [Forecast](./forecast.md) | Graph input worker refill, completion waiting, recovery, and bounded Debug diagnostic comparisons. |
| [Results](./result.md) | Frozen measured comparisons with host/source identity and explicit claim limits. |

The current-source natural-admission matrix is invoked with
`tools/measure/compute/run --virtual-route-matrix <metal|vulkan> [--profile core|full]`.
It is a measurement surface, not a route selector: the public pipeline decides
among `pointwise_callback`, `pointwise_staged` (reported as unsupported when
the current source has no natural distinction), `pointwise_resident`, and
`spatial_window`. The matrix uses page elements 256, core Q=9, frame/capacity
2, and the full pointwise Q set 3/9/257/4096; full Window rows use W=2 for
each Q and W=4 at Q=9/257. Q4096 is retained at 1,048,576 logical elements
and reports a bounded failure instead of shrinking or falling back.

Each case has one fresh cold run, one untimed conditioning run, and 60 warm
synchronous samples in 30 CPU/backend ABBA cycles. The wrapper holds the
accelerator lock and runs three independent process packets; its standalone
EXCLUDE_FROM_ALL executable retains raw rows and validates cardinality,
schema, route keys, semantic identity, proof identity/flags, and ordering
before emitting a three-packet consensus. Core packets must contain exactly
four unique cells; full packets exactly 18, with packet markers 1/2/3. A
route mismatch is `not_comparable`, never a passing fallback, and expected
unsupported/unavailable rows remain in the packet without making enumeration
fail. Comparisons require matching workload/backing/shape/hash, exact observed
route proof/owner/counter/one-Final/publication evidence, and all 60 samples.
Only rows whose timing authority is admitted can publish timing; otherwise
timing fields are unavailable in the aggregate. Published timing summaries
are fieldwise medians of the three packets, and adjacent-Q labels are
published only inside one family/backing/Window key when all three packets
agree; cross-family or cross-backing speedups are invalid.
The report implementation has one physical owner per phase: `report/raw.cpp`
owns the single 134-column schema and raw-row serialization, while
`report/consensus.cpp` owns packet parsing, route-key/semantic gates, timing
hiding, and three-packet median publication. The shared `internal.hpp` only
declares that schema boundary; it does not carry a second report authority.
Every sampled run's status and output are checked after the timed
`Pipeline::run()` wall closes; the check is outside the sample timing and the
first failed sample invalidates the row. The untimed conditioning run is
validated by the same output/hash predicate before warm sampling begins. Raw
CSV route evidence includes the first proof identity, proof-valid count and
identity/flag stability, capability-observation consistency, exact prepared /
executed / Final / publication counts, and an explicit `timing_authority`;
these fields are part of packet semantic agreement.

Metal rows are native-primary. Vulkan rows carry the explicit
`MoltenVK_portability_only` label and are not ranked as native Vulkan.
GPU/kernel and submit spans are emitted as
`unavailable_no_exact_producer` with numeric zero unless an exact producer is
present; no host duration is used to reconstruct them. Observer work is inside
the measured wall and is documented in [Method](./method.md). The matrix is
evidence from the unsealed current source, not a 60-sample performance claim
for the installed Release baseline.

The public behavior being measured is owned by
[Virtual Residency](../../../../node/docs/contracts/compute/residency/README.md).
Ordinary all-staged unary nonresident accelerator Pointwise Q>=2 first probes
`StagedLoop` when exact mapped Host-visible/coherent input/output views exist;
its intended contract is one native recurrence submit, Q GPU epochs, zero
transfer submits/bytes, zero Host epoch submit/service/callbacks, and one
Final/publication/version. A pre-lease structural `BackendUnsupported` cleanly
declines to the legacy route; mutation or any non-capability failure is
terminal. Legacy Pointwise Persistent R2 uses `BackendChunked` two-coordinate
chunks (`ceil(Q/2)` submissions), while unsupported Persistent spatial Window
uses dedicated bounded Window (`Q<=4`) or Stream (`Q>4`) fallback. DeviceVsm
callback packets described by the measurement pages are explicit,
resident/required, or diagnostic routes and must not be read as generic
GPU-generated default execution. The natural StagedLoop Q2/Q3/Q5 fixture is a
passing functional E2E check via
`tools/test/run --fresh compute.pipeline-metal-persistent-sliding`, not a
performance result or a 60-sample timing claim. It verifies one native
submit/handoff, Q GPU epochs, zero Host epoch callbacks/transfers, and one
Final/publication/version. Shared Persistent Q2/Q3/Q5 `BackendChunked` and
direct API Q5/Q9/Q257 checks are separate from public StagedLoop evidence.
Finite workload counts are required by the Persistent Sliding identity
contract; `UINT64_MAX` is a reserved sentinel, and range preflight rejects it
and overflow requests before native owner allocation. This does not attribute
every past IOGPU abort or establish safety for every finite-Q device/driver.
Valid finite-Q direct Metal O(Q) encoding remains blocked on authoritative
native command-capacity/ICB migration.
Generic performance packet and baseline rules remain in
[Performance Method](../method.md).
