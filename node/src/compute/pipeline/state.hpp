#pragma once

#include "../device/state.hpp"
#include "../job/state.hpp"
#include "../program/state.hpp"

#include <rund/compute/abi/model.hpp>
#include <rund/compute/graph/info.hpp>
#include <rund/compute/pipeline/memory.hpp>
#include <rund/compute/pipeline/profile.hpp>
#include <rund/compute/pipeline/shape.hpp>
#include <rund/compute/pipeline/window.hpp>
#include <rund/compute/resource/plan.hpp>
#include <rund/compute/stats.hpp>
#include <rund/compute/status.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <variant>
#include <vector>

namespace rund::compute::detail {

struct PipelinePublicationState;
struct SnapshotStorageState;

struct PipelinePublicationOrdinals final {
  std::array<std::uint32_t, 3u> sources{};
  std::uint32_t count{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t target{std::numeric_limits<std::uint32_t>::max()};
};

enum class PipelinePublishKind : std::uint8_t {
  Terminal,
  Window,
};

struct PipelinePublicationStepOrdinal final {
  std::size_t value{};
};

struct PipelineLogicalOutputOrdinal final {
  std::uint32_t value{};
};

struct PipelinePhysicalOutputOrdinal final {
  std::uint32_t value{};
};

enum class PipelinePhase : unsigned char {
  Ready,
  Running,
  Poisoned,
};

enum class PipelineAccess : unsigned char {
  Read,
  Write,
};

struct BufferClaim final {
  BufferState *buffer{};
  bool write{};
  // Frozen during Pipeline preparation.  Terminal publication can distinguish
  // rollback-owned state writes in the same canonical claim pass without
  // searching every declared state pair for every failed write.
  bool transactional_state{};
  bool gated_publish{};
};

struct PipelineBinding final {
  static constexpr std::uint32_t external =
      std::numeric_limits<std::uint32_t>::max();

  std::shared_ptr<BufferState> buffer{};
  Type type{Type::I32};
  FixedFormat format{};
  std::size_t offset{};
  std::size_t count{};
  std::size_t stride{1u};
  std::size_t element_bytes{};
  std::size_t alignment{};
  std::size_t backing_bytes{};
  ResourceAccess access{ResourceAccess::Read};
  std::uint32_t owner{external};
  bool hidden{};
};

enum class PipelineFill : std::uint8_t {
  None,
  Ordinal,
};

struct PipelineInternal final {
  Type type{Type::U32};
  FixedFormat format{};
  std::size_t count{};
  PipelineFill fill{PipelineFill::None};
};

struct PipelineTerminalPublicationPlan final {
  std::array<std::uint32_t, 3u> sources{};
  std::uint32_t target{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t state{std::numeric_limits<std::uint32_t>::max()};
  PipelinePhysicalOutputOrdinal output{};
  std::uint8_t final{};
};

struct PipelineWindowPublicationPlan final {
  std::uint32_t source{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t count{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t target{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t state{std::numeric_limits<std::uint32_t>::max()};
  PipelinePhysicalOutputOrdinal output{};
  std::uint32_t maximum{};
  std::uint32_t tile{};
};

using PipelinePublicationPlan = std::variant<PipelineTerminalPublicationPlan,
                                             PipelineWindowPublicationPlan>;

struct PipelineMemoryPlan final {
  struct ViewSlot final {
    std::size_t words{};
    std::size_t alignment_words{1u};
    std::size_t owner{};
    std::size_t offset_words{};
  };

  PipelinePlan summary{};
  // Portion of summary.committed_peak_bytes whose owners are retained by the
  // resident publication authority after Pipeline destruction. Zero is an
  // exact value, never a sentinel or request to fall back to peak_bytes.
  std::uint64_t publication_committed_bytes{};
  // Cold schedule authority shared by plan() and prepare(). Resource hazards
  // come from resource::analyze; schedule_barriers is their executable
  // boundary projection plus the shared-workspace reuse frontiers.
  resource::Plan hazards;
  std::vector<std::uint8_t> schedule_barriers;
  // Canonical compact window-state identity for every authored step.  The
  // schedule owns this ordinal assignment; public accelerator accounting and
  // private recurrence admission consume the same frozen projection.
  std::vector<std::uint32_t> window_states;
  std::vector<std::size_t> steps;
  std::vector<std::size_t> owners;
  std::vector<std::size_t> offsets;
  std::vector<std::size_t> chunks;
  std::vector<ViewSlot> view_slots;
  std::vector<std::size_t> view_chunks;
  // Canonical accelerator scratch authority. Bytes and final JobArena slot
  // are frozen once; admission, Buffer materialization, private Jobs, and
  // backend preparation all consume this same descriptor.
  node::accel::detail::KernelScratchLayout scratch;
  std::vector<node::accel::detail::KernelViewLayout> views;
  // Canonical private-Job recurrence owner for every compact route. Planning,
  // materialization, prepared-template counts, and memory admission consume
  // this one mapping instead of independently rediscovering parity reuse.
  std::vector<std::size_t> job_owners;
  // Canonical workspace owner for every compact route. Serial recurrence
  // steps borrow one Program workspace from their first phase; planning and
  // materialization never re-interpret iteration fields independently.
  std::vector<std::size_t> workspace_owners;
  // One sealed CPU preparation plan is the physical authority for the shared
  // serial execution envelope, every immutable route, and every private-Job
  // binding. cpu_storage_by_step indexes cpu_programs and never owns a second
  // Program, mapping, or mutable runner.
  std::vector<std::shared_ptr<ProgramState>> cpu_programs;
  std::vector<CpuGraphStoragePlan> cpu_storage_plans;
  std::vector<CpuRunRoutePlan> cpu_route_plans;
  std::vector<std::size_t> cpu_storage_by_step;
  CpuPreparedArenaPlan cpu_prepared_arena{};
  std::vector<CpuRunRouteSlice> cpu_route_slices;
  std::vector<CpuRunRouteSlice> cpu_alternate_route_slices;
  // CPU private Jobs borrow these exact typed slices from cpu_prepared_arena.
  // Binding and route storage share one cold layout authority; recurrence
  // points at its canonical owner and transactional parity receives a disjoint
  // slice because dense-view staging rewrites owners during preparation.
  std::vector<CpuJobBindingSlice> cpu_job_slices;
  std::vector<CpuJobBindingSlice> cpu_alternate_job_slices;
  // Canonical CPU workspaces and their chunk tables are placement-constructed
  // in the same sealed preparation arena. Recurrent steps borrow their
  // workspace owner's alias and therefore keep an empty slice here.
  std::vector<CpuWorkspaceSlice> cpu_workspace_slices;
  // Exact dense CPU View transfers for each canonical private Job. Reused
  // recurrence entries are empty because they borrow their owner's Job.
  std::vector<CpuViewTransferLayout> cpu_view_layouts;
  // Canonical typed publication authority. Window and Terminal plans contain
  // only their valid coordinates; admission consumes this ordered projection
  // without re-deriving output/input relationships from authored bindings.
  std::vector<PipelinePublicationPlan> publications;
  std::uint64_t publication_fingerprint_hi{};
  std::uint64_t publication_fingerprint_lo{};
  // Frozen accelerator preparation admission. The public plan owns this
  // immutable descriptor until bind hands it to PipelineState; no backend or
  // native owner exists while this value is computed.
  node::accel::detail::PreparedKernelPipelineReservation accel_preparation{};
};

// A nested window keeps one compact table of reusable routes.  Route entries
// are not execution occurrences: seed has one ordinal-specific route per
// outer window, action has one route per inner parity occurrence, and fold has
// the three seed/first/second outer-state transitions.  The warm executor
// interprets the two independent bounds without materializing their product.
enum class PipelineRoute : std::uint8_t {
  Ordinary,
  NestedSeed,
  NestedAction,
  NestedFold,
};

struct PipelineBuildStep final {
  std::shared_ptr<ProgramState> program;
  std::vector<PipelineBinding> inputs;
  std::vector<PipelineBinding> outputs;
  std::uint32_t logical_step{};
  std::uint32_t iteration{};
  std::uint32_t iteration_bound{1u};
  std::size_t window_max{};
  std::size_t window_tile{};
  std::size_t window_terminal{NoWindowTerminal};
  std::uint32_t window_expected{1u};
  std::uint16_t nested{};
  PipelineRoute route{PipelineRoute::Ordinary};
  // True only when repeat(...) retains this occurrence in caller-owned
  // iteration history. It prevents a terminal-only recurrence lowering from
  // eliding the authored intermediate writes.
  bool writes_each_iteration{};
};

struct PipelineBuildNestedWindow final {
  PipelineBinding count;
  std::size_t begin{};
  std::size_t end{};
  std::size_t seed_first{};
  std::size_t seed_count{};
  std::size_t action_first{};
  std::size_t action_count{};
  std::size_t fold_first{};
  // Exact leading Fold-output prefix that is also outer recurrent state.
  // Append-only window outputs follow this prefix and never enter a bank seal.
  std::size_t recurrent_output_count{};
  std::size_t maximum{};
  std::size_t tile{};
  std::size_t terminal{NoWindowTerminal};
  std::uint32_t expected{1u};
};

struct PipelineBuildStatePair final {
  PipelineBinding published;
  PipelineBinding pending;
};

// Recurrent state and append-only window output both compute into private
// storage. Terminal publication is gated on complete Pipeline success. Window
// publication is count-gated after each successful Fold and is intentionally
// non-rollback: a later failure poisons a destination that may contain an
// already published prefix.
struct PipelineBuildPublicationEdge final {
  PipelineBinding source;
  PipelineBinding target;
  PipelinePublicationStepOrdinal step{};
  PipelineLogicalOutputOrdinal output{};
};

struct PipelineBuildTerminalPublication final {
  PipelineBuildPublicationEdge edge;
};

struct PipelineBuildWindowPublication final {
  PipelineBuildPublicationEdge edge;
  PipelineBinding count;
  std::size_t maximum{};
  std::size_t tile{};
};

using PipelineBuildPublication = std::variant<PipelineBuildTerminalPublication,
                                              PipelineBuildWindowPublication>;

[[nodiscard]] inline const PipelineBuildPublicationEdge &
pipeline_publication_edge(const PipelineBuildPublication &publication) {
  return std::visit(
      [](const auto &typed) -> const PipelineBuildPublicationEdge & {
        return typed.edge;
      },
      publication);
}

[[nodiscard]] inline PipelineBuildPublicationEdge &
pipeline_publication_edge(PipelineBuildPublication &publication) {
  return std::visit(
      [](auto &typed) -> PipelineBuildPublicationEdge & { return typed.edge; },
      publication);
}

struct PipelineBuildState final {
  std::shared_ptr<DeviceState> device;
  std::vector<PipelineBuildStep> steps;
  std::vector<PipelineBuildStatePair> state_pairs;
  std::vector<PipelineBuildPublication> publications;
  std::vector<PipelineInternal> internals;
  std::vector<PipelineBuildNestedWindow> nested_windows;
  std::shared_ptr<const PipelineMemoryPlan> memory;
  std::shared_ptr<StateSnapshotState> seed;
  std::shared_ptr<SnapshotStorageState> storage_seed;
  std::shared_ptr<PipelinePublicationState> device_seed;
  std::size_t binding_count{};
  std::size_t logical_step_count{};
  std::uint32_t sealed_repetitions{1u};
  PipelineProfile profile{PipelineProfile::None};
  std::uint64_t budget{};
  bool has_budget{};
  bool commit{};
  bool sealed{};
  bool sealed_repetitions_configured{};
  Reason failure{Reason::Ok};
};

struct PipelineProfileState final {
  std::vector<PipelineStepProfile> steps;
  std::vector<std::uint64_t> started_ns;
  std::vector<bool> started;
  std::uint64_t instrumentation_command_count{};
  std::uint64_t instrumentation_byte_count{};
};

struct PipelineResource final {
  static constexpr std::uint32_t no_output =
      std::numeric_limits<std::uint32_t>::max();

  std::shared_ptr<BufferState> buffer;
  Type type{Type::I32};
  FixedFormat format{};
  std::size_t count{};
  std::size_t bytes{};
  std::size_t offset{};
  std::size_t stride{1u};
  std::size_t element_bytes{};
  std::size_t alignment{};
  std::uint32_t output{no_output};
  std::uint32_t first_write{resource::NoNode};
  bool owned{};
  bool terminal_publish{};
};

struct PipelineOutputState final {
  std::uint64_t generation{};
  std::uint64_t hash{};
  std::uint32_t resource{};
  bool observed{};
};

struct PipelineStep final {
  std::shared_ptr<ProgramState> program;
  std::shared_ptr<JobState> job;
  std::shared_ptr<JobState> alternate_job;
  std::uint32_t logical_step{};
  std::uint32_t iteration{};
  std::uint32_t iteration_bound{1u};
  // Binding descriptors and resource ordinals are cold planning authority and
  // are not retained after the two private Jobs/native streams are frozen.
  // Warm execution needs only whether this step can make a write observable.
  bool writes : 1 {};
  bool writes_each_iteration : 1 {};
  PipelineRoute route{PipelineRoute::Ordinary};
  // One-based index into PipelineState::windows.
  std::uint16_t window{};
};

static_assert(sizeof(PipelineStep) <=
              sizeof(std::shared_ptr<ProgramState>) * 4u);

struct PipelineWindow final {
  static constexpr std::uint32_t seed = 0u;
  static constexpr std::uint32_t first = 1u;
  static constexpr std::uint32_t second = 2u;

  std::shared_ptr<BufferState> count;
  std::size_t count_offset{};
  std::size_t first_step{};
  // Leading output/input bank prefix sealed when no resident work executes.
  std::uint32_t recurrent_output_count{};
  std::uint32_t maximum{};
  std::uint32_t tile{};
  std::uint32_t terminal{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t terminal_output{};
  std::uint32_t expected{1u};
  std::uint32_t current{seed};
  std::size_t begin{};
  std::size_t end{};
  std::size_t seed_first{};
  std::size_t seed_count{};
  std::size_t action_first{};
  std::size_t action_count{};
  std::size_t fold_first{};
  bool nested{};
  bool stopped{};
};

struct PipelineStatePair final {
  std::shared_ptr<BufferState> first;
  std::shared_ptr<BufferState> second;
  Type type{Type::I32};
  FixedFormat format{};
  std::size_t count{};
  std::size_t bytes{};
};

// Backend-ready execution descriptor derived from PipelinePublicationPlan.
// Its flat shape is a warm-path encoding, never a second admission authority.
struct PipelinePublish final {
  std::shared_ptr<BufferState> source;
  std::shared_ptr<BufferState> target;
  std::shared_ptr<BufferState> resident_count;
  Type type{Type::I32};
  FixedFormat format{};
  std::size_t source_offset{};
  std::size_t target_offset{};
  std::size_t count{};
  std::size_t target_stride{1u};
  std::size_t element_bytes{};
  std::uint16_t window{};
  std::uint16_t output{};
  PipelinePublishKind kind{PipelinePublishKind::Terminal};
  std::uint8_t final{};
  std::uint32_t state{std::numeric_limits<std::uint32_t>::max()};
  std::size_t resident_count_offset{};
  std::uint32_t maximum{};
  std::uint32_t tile{};
  PipelinePublicationOrdinals ordinals{};
};

struct PipelineSnapshotField final {
  Type type{Type::I32};
  FixedFormat format{};
  std::size_t count{};
  std::size_t offset{};
  std::size_t bytes{};
  // Produced by the copy/download that materializes this field.  Snapshot
  // identity consumes the retained leaf hash instead of walking the complete
  // payload a second time immediately after that copy.
  std::uint64_t payload_hash{};
};

struct StateSnapshotState final {
  graph::Fingerprint fingerprint{};
  std::vector<PipelineSnapshotField> fields;
  // Snapshot construction overwrites the complete payload exactly once.
  // Keep raw owned storage so allocation does not first zero every byte.
  std::unique_ptr<std::byte[]> bytes;
  std::size_t byte_count{};
  std::uint64_t generation{};
  std::uint64_t hash{};
};

struct SnapshotStorageState final {
  mutable std::mutex gate;
  std::array<StateSnapshotState, 2u> banks;
  std::size_t byte_capacity{};
  std::size_t field_capacity{};
  std::uint8_t active{};
  bool valid{};
};

// Resident checkpoint authority shared by Pipeline and LatestDeviceState.
// Frozen owners and schema survive Pipeline destruction, while the publication
// selector changes only in the terminal Device-claim critical section.
struct PipelinePublicationState final {
  std::shared_ptr<DeviceState> device;
  // Declared before publication resources so reverse destruction releases the
  // aggregate Device charge only after every resident owner is gone.
  storage::Reservation publication_memory;
  std::vector<PipelineStatePair> state_pairs;
  graph::Fingerprint fingerprint{};
  mutable std::mutex gate;
  std::uint64_t generation{};
  // Changes whenever this authority publishes different resident payload,
  // including a restore whose generation/parity happen to stay unchanged.
  std::uint64_t payload_epoch{};
  std::uint8_t parity{};
  bool device_lost{};
  // At most one Pipeline may execute against shared pair owners. The attempt
  // reservation freezes selector identity without holding this mutex across
  // an asynchronous backend submission.
  bool attempt_active{};
};

struct PipelineDependency final {
  std::uint32_t before{};
  std::uint32_t after{};
  std::uint32_t resource{};
  PipelineAccess before_access{PipelineAccess::Read};
  PipelineAccess after_access{PipelineAccess::Read};
};

struct PipelineState final {
  std::shared_ptr<DeviceState> device;
  // Declared before all Pipeline-private owners so reverse destruction keeps
  // the aggregate Device charge live until those owners are gone.
  storage::Reservation private_memory;
  std::shared_ptr<PipelinePublicationState> publication;
  // Canonical owner of the one CPU prepared mapping. Jobs and Program storage
  // retain lifetime references to this same control block, but planning,
  // observation, and destruction read this Pipeline-owned authority.
  std::shared_ptr<CpuPreparedArena> cpu_prepared_arena;
  std::vector<PipelineStep> steps;
  std::vector<PipelineWindow> windows;
  // Sealed rank of physical window steps in each prefix. Empty for a Pipeline
  // without resident windows; otherwise size is steps.size() + 1.
  std::vector<std::uint16_t> window_rank;
  std::vector<PipelineResource> resources;
  std::vector<std::shared_ptr<BufferState>> prepared_buffers;
  std::vector<std::shared_ptr<CpuGraphStorage>> cpu_storage;
  std::vector<BufferClaim> claims;
  std::vector<BufferClaim> alternate_claims;
  std::vector<PipelinePublish> publications;
  std::vector<PipelineOutputState> outputs;
  // Lookup-only permutation of outputs, sorted by Buffer owner address. The
  // canonical output vector remains in resource order and owns hashing order.
  std::vector<std::uint32_t> output_lookup;
  std::vector<PipelineDependency> dependencies;
  std::vector<std::uint8_t> barriers;
  std::unique_ptr<PipelineProfileState> profile;
  // Sole mutable authority for Program-template publication across primary
  // and transactional-alternate native streams. Its limit is copied from the
  // frozen memory plan before private Jobs are materialized.
  node::accel::detail::PreparedKernelTemplateRegistry accel_templates;
  node::accel::detail::PreparedKernelPipeline prepared;
  node::accel::detail::PreparedKernelPipeline alternate_prepared;
  PipelinePlan plan{};
  mutable std::mutex gate;
  PipelinePhase phase{PipelinePhase::Ready};
  Stats stats{};
  CheckpointStats checkpoint_stats{};
  std::uint64_t status_entry_count{};
  std::size_t logical_step_count{};
  // One physical execution represents this many input-sealed evaluations at
  // one terminal observation after the cold temporal proof succeeds. It never
  // denotes this many public run()/generation transitions.
  std::uint32_t sealed_repetitions{1u};
  // Sealed during preparation. Warm accelerator execution uses this immutable
  // count instead of walking private Jobs to rediscover the active subset.
  std::uint32_t active_step_count{};
  std::uint64_t attempt_generation{};
  std::uint64_t native_generation{};
  std::uint64_t observation_generation{};
  std::uint64_t observation_payload_epoch{};
  std::uint8_t attempt_parity{};
  std::uint8_t native_parity{};
  std::uint8_t observation_parity{};
  Reason failure{Reason::Ok};
  std::size_t verified{};
  bool failure_step_known{};
  bool writes_possible{};
  bool backend_submitted{};
  bool transactional{};
  bool preparing{true};
  bool observation_identity_valid{};
  // A submitted failure whose native generation control could not be rebased
  // cannot safely execute again: the next completion identity is unknowable.
  bool control_poisoned{};
  std::uint64_t frame_current{};
  std::uint64_t frame_peak{};
  std::uint64_t frame_bytes{};
  std::uint64_t frame_reused{};
  std::uint64_t frame_budget{};
  std::uint64_t read_transfer_peak{};
  std::uint64_t read_transfer_bytes{};
  std::uint64_t read_staging_peak{};
  std::uint64_t read_staging_bytes{};
  std::uint64_t read_staging_reused{};
  std::uint64_t read_staging_budget{};
  std::size_t unobserved_outputs{};
};

} // namespace rund::compute::detail
