#include "local.hpp"

#include "src/compute/pipeline/residency/planner.hpp"
#include "src/compute/virtual/graph/reduce/wavefront.hpp"

#include <array>
#include <memory>
#include <utility>

namespace rund_node_test_pipeline_residency {

int CheckGraphWavefront() {
  using rund::compute::Status;
  using rund::compute::detail::Type;
  using namespace rund::compute::detail;
  using namespace residency;
  using graph_reduce::WavefrontCoordinate;

  const TiledGraphPlanInput input{
      .page_count = 5u,
      .requested_frames = 2u,
      .max_frames = 2u,
      .prefetch_distance = 2u,
      .resources = {{.resource = 1u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 4u * 64u + 40u,
                     .kind = GraphResourceKind::ExternalInput,
                     .persistence = ResourcePersistence::Backing},
                    {.resource = 2u,
                     .type = Type::U64,
                     .page_bytes = 128u,
                     .logical_bytes = 4u * 128u + 72u,
                     .kind = GraphResourceKind::ExternalInput,
                     .persistence = ResourcePersistence::Backing},
                    {.resource = 3u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 4u * 64u + 40u,
                     .kind = GraphResourceKind::Internal,
                     .persistence = ResourcePersistence::Transient},
                    {.resource = 4u,
                     .type = Type::U64,
                     .page_bytes = 128u,
                     .logical_bytes = 4u * 128u + 72u,
                     .kind = GraphResourceKind::Internal,
                     .persistence = ResourcePersistence::Transient},
                    {.resource = 5u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 4u * 64u + 40u,
                     .kind = GraphResourceKind::ExternalOutput,
                     .persistence = ResourcePersistence::Transient}},
      .stages = {{.node = 1u,
                  .ports = {{.resource = 1u, .access = Access::Read},
                            {.resource = 3u, .access = Access::Write}}},
                 {.node = 2u,
                  .ports = {{.resource = 2u, .access = Access::Read},
                            {.resource = 4u, .access = Access::Write}}},
                 {.node = 3u,
                  .ports = {{.resource = 3u, .access = Access::Read},
                            {.resource = 4u, .access = Access::Read},
                            {.resource = 5u, .access = Access::Write}}}},
  };
  PlanResult planned = PlanResidency(input);
  if (!planned) {
    return 1;
  }
  auto owner = std::make_shared<ResidencyPlan>(std::move(planned.plan));
  const std::array<std::uint64_t, 5u> logical_bytes{
      4u * 64u + 40u, 4u * 128u + 72u, 4u * 64u + 40u, 4u * 128u + 72u,
      4u * 64u + 40u};
  TiledGraphInvocation invocation{};
  graph_reduce::Wavefront wavefront{};
  if (!owner->tiled_graph().active(5u, logical_bytes, invocation) ||
      !wavefront.reset(invocation, owner->tiled_graph()) ||
      wavefront.retained_cells() != graph_reduce::WavefrontCellCapacity ||
      !wavefront.admit(0u) || !wavefront.admit(1u)) {
    return 2;
  }

  // Reverse completion: later independent stage/input 1 reaches HostReady and
  // exact device readiness while stage 0 I/O is still blocked. The join must
  // not overtake either sealed producer.
  WavefrontCoordinate selected{};
  WavefrontCoordinate pending{};
  std::uint32_t pending_resource = 0u;
  if (!wavefront.forecast(pending, pending_resource) || pending.batch != 0u ||
      pending.stage != 0u || pending_resource != 1u ||
      wavefront.promote(0u, pending) ||
      !wavefront.forecast_terminal(0u, 1u, 2u, Status::success()) ||
      !wavefront.promote(0u, pending) || pending.stage != 1u ||
      !wavefront.device_ready(0u, 1u) || !wavefront.select(selected) ||
      selected.batch != 0u || selected.stage != 1u ||
      !wavefront.dispatch(selected) || !wavefront.terminal(selected) ||
      wavefront.select(selected)) {
    return 3;
  }

  // Forecast terminal is HostReady, not a fabricated Promote/Native fact.
  if (!wavefront.forecast_terminal(0u, 0u, 1u, Status::success()) ||
      wavefront.select(selected) || !wavefront.device_ready(0u, 0u) ||
      !wavefront.select(selected) || selected.stage != 0u ||
      !wavefront.dispatch(selected) || !wavefront.terminal(selected) ||
      !wavefront.select(selected) || selected.stage != 2u ||
      !wavefront.dispatch(selected) || !wavefront.terminal(selected)) {
    return 4;
  }

  // The opposite bank may finish both independent producers while bank 0's
  // output is still awaiting Drain/release. Its join has a planner-sealed
  // ReleaseComplete edge and cannot reuse that physical Output frame merely
  // because the producer Dispatch terminal exists.
  if (!wavefront.forecast_terminal(1u, 1u, 2u, Status::success()) ||
      !wavefront.device_ready(1u, 1u) || !wavefront.select(selected) ||
      selected.batch != 1u || selected.stage != 1u ||
      !wavefront.dispatch(selected) || !wavefront.terminal(selected) ||
      wavefront.select(selected) ||
      !wavefront.forecast_terminal(1u, 0u, 1u, Status::success()) ||
      !wavefront.device_ready(1u, 0u) || !wavefront.select(selected) ||
      selected.stage != 0u || !wavefront.dispatch(selected) ||
      !wavefront.terminal(selected) || wavefront.select(selected) ||
      !wavefront.release(0u) || !wavefront.select(selected) ||
      selected.stage != 2u || !wavefront.dispatch(selected) ||
      !wavefront.terminal(selected) || !wavefront.release(1u)) {
    return 5;
  }

  // Reusing bank 0 keeps a constant 2*stage bound. With both independent
  // inputs ready, the deterministic key is stage, coordinate, resource. The
  // last batch projects the exact K=2 tail [4,5).
  if (!wavefront.admit(2u) ||
      !wavefront.forecast_terminal(2u, 1u, 2u, Status::success()) ||
      !wavefront.device_ready(2u, 1u) ||
      !wavefront.forecast_terminal(2u, 0u, 1u, Status::success()) ||
      !wavefront.device_ready(2u, 0u) || !wavefront.select(selected) ||
      selected.stage != 0u || selected.first_page != 4u ||
      selected.page_count != 1u || !wavefront.dispatch(selected) ||
      !wavefront.terminal(selected) || !wavefront.select(selected) ||
      selected.stage != 1u || !wavefront.dispatch(selected) ||
      !wavefront.terminal(selected) || !wavefront.select(selected) ||
      selected.stage != 2u || !wavefront.dispatch(selected) ||
      !wavefront.terminal(selected) || !wavefront.release(2u)) {
    return 6;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency
