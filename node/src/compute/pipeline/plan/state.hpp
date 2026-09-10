#pragma once

#include "../state/plan.hpp"

namespace rund::compute::detail {

struct PipelineMemoryPlan final {
  struct ViewSlot final {
    std::size_t words{};
    std::size_t alignment_words{1u};
    std::size_t owner{};
    std::size_t offset_words{};
  };

  PipelinePlan summary{};
  // Declared before every owner-bearing cold vector. Reverse destruction must
  // drop BufferState aliases before the Pool may release physical regions or
  // refund their Device admission.
  PipelineResidencyPlan residency{};
  // Immutable declaration snapshot and canonical resource coordinates. Once
  // plan() publishes this object, prepare() never reads authored bindings
  // again; it materializes owners and Jobs from these records only.
  std::shared_ptr<const PipelineBuildSnapshot> frozen;
  std::vector<PipelineStepResourcePlan> step_resources;
  std::vector<PipelineResolvedResourcePlan> resources;
  std::vector<PipelineStatePairResourcePlan> state_pair_resources;
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
  // Canonical workspace state for every compact route. Serial recurrence
  // steps borrow one Program workspace from their phase root; absent routes
  // remain explicit through planning and materialization.
  std::vector<PipelineWorkspaceRoute> workspace_routes;
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
  // Exact count View for each zero-based window state. This exists even when a
  // window has no append-only publication, so count gating never falls back to
  // authored bindings or a canonical-only Buffer pointer.
  std::vector<PipelineWindowControl> window_controls;
  std::uint64_t publication_fingerprint_hi{};
  std::uint64_t publication_fingerprint_lo{};
  // Frozen accelerator preparation admission. The public plan owns this
  // immutable descriptor until bind hands it to PipelineState; no backend or
  // native owner exists while this value is computed.
  node::accel::detail::PreparedKernelPipelineReservation accel_preparation{};
};

} // namespace rund::compute::detail
