# Residency Pool

This page owns physical arenas, eligible regions, working-set formulas,
cross-layout lending, and memory accounting. Pools own mechanisms and bytes,
never replacement policy.

Let page payload elements be `E`, input/output widths be `b_i,b_o`, logical
count be `N`, and physical page bytes be `G_i,G_o`:

```text
G_i   = checked E_i * b_i
G_o   = checked E_o * b_o
P     = ceil(N / payload_elements)
K_cpu = min(P, floor(host_resident_bytes / (2 * (G_i + G_o))))
K_acc = min(P, floor(device_resident_bytes / (2 * (G_i + G_o))),
               floor(host_resident_bytes / (2 * (G_i + G_o))))
D     = min(P, floor(floor(host_resident_bytes / 2) / (G_i + G_o)))
r_0   = floor(host_resident_bytes / 2) - D * (G_i + G_o)
X_i   = min(P-D, floor(r_0 / G_i))
r_1   = r_0 - X_i * G_i
X_o   = min(P-D, floor(r_1 / G_o))
H_acc = D + X_i
O_acc = D + X_o
K     = K_cpu on CPU, K_acc on an accelerator
H     = K_cpu on CPU, H_acc on an accelerator, with H >= K
O     = K_cpu on CPU, O_acc on an accelerator, with O >= K
Q     = ceil(P / K)
L     = checked N * (b_i + b_o)
R_cpu = checked 2 * K_cpu * (G_i + G_o)
R_acc = checked 2 * K_acc * (G_i + G_o)
S_acc = checked 2 * (H_acc * G_i + O_acc * G_o)
```

`D` first gives Forecast and Persist the same complete executable depth.
Because the rings are independently tagged, an unpaired per-bank remainder is
then assigned to the reusable input cache before the transient output ring;
any bytes too small for another input page may still extend output. This uses
the admitted Host budget without manufacturing a second cache policy or
reducing either ring below `K`.

CPU's two banks are authoritative Host execution frames. Accelerators own two
Device input/output banks, `2H` Host input-cache frames, and `2O` Host output
staging frames. Backing reads target either the exact Host input later used by
private upload, or—only for authenticated Direct equal-capacity coherent
supply—the exact HostVisible Device Input binding under the compound execution
token;
prefetch workers retain metadata only. Coherent output does not remove retained
fallback capacity because capability absence must remain executable. Coherent
equal-capacity input likewise keeps its already admitted Host frames; current
work removes the physical copy, not cold fallback capacity.
The Prefetcher implementation is split by lifecycle: `prefetch.cpp` owns cold
configuration and destruction, while `prefetch/submit.cpp`, `work.cpp`,
`observe.cpp`, and `cancel.cpp` own request publication, worker I/O, receipt
observation, and Authority/alias teardown respectively. `support.cpp` is the
single timing and alias-match formula owner. The declarations-only local seam
stores no worker state or second completion token.

The execution Plan names the complete `H`-frame Host input and `O`-frame Host
output regions for each bank, not reconstructed K-prefixes. A projected epoch
still binds at most `K` pages. Authority searches the exact H-region for hits, then an Empty frame,
then an authenticated consumed victim, while its Device target remains the
fixed K local. Thus `H>K` is an ordinary bounded cache shape rather than a
reason to reject recurrent execution. Drain reserves an exact O-ring slot
before native mutation, so a slow Persist releases the K-sized Device bank
without allowing Host staging overwrite. The retained journal is O(H+O+K)
with `H,O<=UseCapacity`; it never stores Q page rows.

For adjacent Window pages, the input projection intersects the two exact halo
footprints. The next materialization copies the overlap from one authenticated
predecessor frame and reads only the unique suffix before applying terminal
Clamp/Clip fill. CPU rolling execution and the recurrent compound owner retain
this O(1) predecessor seed across epoch boundaries, so a monotonic run reads
each logical input byte once. Accelerator rolling prefetch does not yet carry
that cross-epoch seed because its asynchronous lifetime has a separate owner;
it remains a documented partial boundary rather than borrowing an
unauthenticated pointer.

## Graph Arenas

For unique graph physical classes `G_j`, control bytes `G_c`, requested
capacity `K`, and canonical arena capacities `C_j`:

```text
B_K      = checked 2 * K * (G_c + sum_j G_j)
R_global = checked 2 * sum_j (C_j * G_j)
```

`B_K` derives executable capacity. `R_global` charges each actual extent once.
A borrower may retain a `C>K` owner while executing explicit K-prefix regions;
its plan publishes K and Device memory observes the shared C owner once.

The extent key is committed per-bank storage bin plus execution tier, separate
from semantic view identity. Input/Output arenas additionally carry the
backend-neutral source-private `HostVisiblePreferred` intent; this prevents a
VSM bank that requires a coherent Host capability from borrowing an ordinary
arena, and prevents an ordinary arena from inheriting the VSM mapping policy.
CPU and Metal adapters ignore the memory hint; Vulkan consumes it only when an
actual coherent Device-local memory type exists. A compatible Graph Pool may expose typed views
with different role, page geometry, Type, or FixedFormat over the same native
bytes. A changed scalar width/count must be sealed into a new typed
AccelBuffer capability through the existing context view-admission owner;
copying the allocation's old typed capability would contradict the new
BufferState view. The backend-neutral `project_buffer_view` callback performs
that cold projection without native allocation or transfer, while the
physical owner, resident ID, byte extent, and allocation charge stay shared.
Same-shape views retain the existing capability. Native binding checks still
require the sealed scalar width, count, alignment, and authenticated extent.
GraphResident admission authenticates the original arena and its borrowed view
with their own page units: `bank_bytes / arena_page_bytes == arena_capacity`
and `bank_bytes / view_page_bytes == view_capacity`, with exact divisibility.
Comparing those two capacities directly is invalid when reblocking changes
page width; both describe the same authenticated byte extent.
Semantic cache identity remains distinct. Page-geometry or role changes
receive distinct Authority rows; type-only views of identical geometry may use
canonical rows. Run admission activates one coordinate view per extent,
evicts only clean rows of another view, and rejects dirty rows as Capacity and
in-flight rows as Busy.

Pool release applies the same Authority owner gate: the requested row and any
equal nonzero `PhysicalArena` extent remain non-releasable while an active
journal reference or alias claim exists, including active cycle membership.
Evidence/progress is not a release owner. Extent-zero covers only ordinary
exact-region reuse; arbitrary `BufferState`/resident alias reuse is not
supported. Graph extent selection therefore requires an existing nonzero
extent/view pair. An ordinary extent-zero candidate is skipped, and a fresh
Graph arena receives its own identities through the existing allocation
transaction; a loan never rewrites the identity of an ordinary arena or its
live rows.

Authority hit search spans the two full C banks. Execution uses only its two
explicit K regions; bank adjacency is never reconstructed. Relocation into the
selected prefix follows dependency order. A permutation cycle requires one
clean unretained scratch frame outside pending sources/targets; a fully live
`C=K` cycle without scratch is Capacity. All sources, targets, and scratch stay
in one transaction.

The public U64 Graph product executes 128-byte and 64-byte views over the same
raw Input and Intermediate owners and switches back with exact CPU, Metal, and
Vulkan output and fixed runD-owned warm memory. Cross-type and cross-role execution are
structural owner/Authority evidence, not public kernel evidence. Direct Stream
layouts lend only Input.

Different committed bins, global dirty-live victim reclamation, smaller-to-
larger relocation, split/coalesce/growth, and one runtime victim choice across
incompatible native buffers remain partial.

## Accounting

Each new arena's exact backend allocation requirement is reserved once from
the Device Pipeline budget until the last owner dies. Borrowing views and
Pipelines do not reserve or report it again. CPU uses the Host allocation;
Metal uses `allocatedSize`; Vulkan planning uses
`VkMemoryRequirements::size`. Accelerator Host frames are a separate exact
Pool charge. No residency-specific allocator or memory counter exists.

## Implementation Ownership

The physical pool implementation is split by lifecycle authority:

```text
node/src/compute/device/residency/pool.cpp
    live Pool execution configure/submit/wait and graph-owner lookup
node/src/compute/device/residency/pool/footprint.cpp
    checked host/graph footprint projections and retained-host accounting
node/src/compute/device/residency/pool/lifecycle.cpp
    PhysicalArena/Pool destruction, Authority release, and refund teardown
node/src/compute/device/residency/pool/acquire.cpp
    Registry gate, common validation, live-owner keepalive, cache lookup, and
    graph/ordinary projection dispatch
node/src/compute/device/residency/pool/acquire/common.hpp/.cpp
    shared physical reservation/arena/buffer/frame-registration transaction,
    admission commit, Host accounting, publication, and rollback
node/src/compute/device/residency/pool/acquire/graph.cpp
    Graph physical-class projection, typed views, and semantic region wiring
node/src/compute/device/residency/pool/acquire/ordinary.cpp
    ordinary Input-arena projection and named execution-region wiring
node/src/compute/device/residency/pool/internal.hpp/.cpp
    only the frame-role and graph-class identity seams shared by acquisition
    and footprint validation
```

`Registry::acquire` remains the only public atomic admission boundary. The
private transaction owns the shared reservation, registration, accounting,
commit, publication, and rollback order; Graph and ordinary files only project
their distinct physical owners and semantic regions into that transaction.
