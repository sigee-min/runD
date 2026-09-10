#include "src/accel/kernel/prepared/interface/api.hpp"
#include "src/accel/kernel/prepared/model.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

#include "hooks.hpp"
#include "route.hpp"

namespace node_accel_contract {

using rund::node::accel::detail::BackendRun;

namespace {

[[nodiscard]] bool MatchBackendTemplateGroup(const BackendRun &left,
                                             const BackendRun &right) noexcept {
  return left.original_dispatch_count == right.original_dispatch_count;
}

[[nodiscard]] bool
AsymmetricBackendTemplateGroup(const BackendRun &left,
                               const BackendRun &right) noexcept {
  return left.original_dispatch_count <= right.original_dispatch_count;
}

} // namespace

[[nodiscard]] bool BackendTemplateRouteDemandIsExactAndFrozen() {
  using namespace rund::node::accel::detail;
  BackendTemplateRouteDemand scalar{
      .owner_count = 7u, .route_copies = 1u, .capacity = 7u};
  if (!BackendTemplateRouteDemandForContract(2u, 2u, scalar) ||
      scalar.owner_count != 2u || scalar.route_copies != 2u ||
      scalar.capacity != 4u || !scalar.valid()) {
    return false;
  }
  const BackendTemplateRouteDemand frozen = scalar;
  if (BackendTemplateRouteDemandForContract(
          std::numeric_limits<std::uint32_t>::max(), 2u, scalar) ||
      scalar.owner_count != frozen.owner_count ||
      scalar.route_copies != frozen.route_copies ||
      scalar.capacity != frozen.capacity) {
    return false;
  }
  for (const std::uint64_t illegal_copies : {0u, 3u}) {
    if (BackendTemplateRouteDemandForContract(2u, illegal_copies, scalar) ||
        scalar.owner_count != frozen.owner_count ||
        scalar.route_copies != frozen.route_copies ||
        scalar.capacity != frozen.capacity) {
      return false;
    }
  }

  const BackendOps grouped_ops{
      .same_pipeline_template = MatchBackendTemplateGroup,
  };
  const auto route = [&](const std::uint64_t group) {
    auto state = std::make_shared<prepared::RunState>();
    state->mode = KernelPreparationMode::PipelinePrivate;
    state->bound.run.ops = &grouped_ops;
    state->bound.run.original_dispatch_count = group;
    return state;
  };
  const std::shared_ptr<prepared::RunState> first = route(11u);
  const std::shared_ptr<prepared::RunState> second = route(11u);
  const std::shared_ptr<prepared::RunState> other = route(29u);
  // A route already materialized by an earlier attempt remains one semantic
  // owner in the group; it is neither omitted nor counted twice.
  first->backend = std::make_shared<int>(7);
  std::array<PreparedKernelRun, 4u> owners{
      PreparedKernelRun{.owner = first, .ok = true, .reason = "ok"},
      PreparedKernelRun{.owner = second, .ok = true, .reason = "ok"},
      PreparedKernelRun{.owner = first, .ok = true, .reason = "ok"},
      PreparedKernelRun{.owner = other, .ok = true, .reason = "ok"},
  };
  const std::array<const PreparedKernelRun *, 4u> runs{&owners[0], &owners[1],
                                                       &owners[2], &owners[3]};
  std::array<BackendTemplateRouteDemand, 4u> demands{};
  std::uint64_t unique_routes = 0u;
  std::uint64_t templates = 0u;
  if (!PlanBackendTemplateRouteDemandsForContract(runs, 2u, demands,
                                                  unique_routes, templates) ||
      unique_routes != 3u || templates != 2u) {
    return false;
  }
  for (const std::size_t index : {0u, 1u, 2u}) {
    if (demands[index].owner_count != 2u || demands[index].route_copies != 2u ||
        demands[index].capacity != 4u || !demands[index].valid()) {
      return false;
    }
  }
  if (demands[3].owner_count != 1u || demands[3].route_copies != 2u ||
      demands[3].capacity != 2u || !demands[3].valid() ||
      !BackendPreparationCursorLifecycleForContract(first->bound.run,
                                                    demands[0])) {
    return false;
  }

  // The complete prepass is transactional for an illegal generation stride:
  // no demand or cardinality escapes the rejected plan.
  const auto legal_demands = demands;
  unique_routes = 43u;
  templates = 47u;
  if (PlanBackendTemplateRouteDemandsForContract(runs, 3u, demands,
                                                 unique_routes, templates) ||
      demands[0].capacity != legal_demands[0].capacity ||
      demands[1].capacity != legal_demands[1].capacity ||
      demands[2].capacity != legal_demands[2].capacity ||
      demands[3].capacity != legal_demands[3].capacity ||
      unique_routes != 43u || templates != 47u) {
    return false;
  }

  // Backend equality is an equivalence authority. A predicate that groups a
  // pair in only one direction is rejected before any cursor/native reserve.
  const BackendOps asymmetric_ops{
      .same_pipeline_template = AsymmetricBackendTemplateGroup,
  };
  const auto asymmetric_route = [&](const std::uint64_t group) {
    auto state = std::make_shared<prepared::RunState>();
    state->mode = KernelPreparationMode::PipelinePrivate;
    state->bound.run.ops = &asymmetric_ops;
    state->bound.run.original_dispatch_count = group;
    return state;
  };
  const auto asymmetric_first = asymmetric_route(1u);
  const auto asymmetric_second = asymmetric_route(2u);
  std::array<PreparedKernelRun, 2u> asymmetric_owners{
      PreparedKernelRun{.owner = asymmetric_first, .ok = true, .reason = "ok"},
      PreparedKernelRun{.owner = asymmetric_second, .ok = true, .reason = "ok"},
  };
  const std::array<const PreparedKernelRun *, 2u> asymmetric_runs{
      &asymmetric_owners[0], &asymmetric_owners[1]};
  std::array<BackendTemplateRouteDemand, 2u> rejected_demands{
      BackendTemplateRouteDemand{
          .owner_count = 3u, .route_copies = 1u, .capacity = 3u},
      BackendTemplateRouteDemand{
          .owner_count = 5u, .route_copies = 1u, .capacity = 5u},
  };
  const auto rejected_before = rejected_demands;
  unique_routes = 31u;
  templates = 37u;
  return !PlanBackendTemplateRouteDemandsForContract(
             asymmetric_runs, 1u, rejected_demands, unique_routes, templates) &&
         rejected_demands[0].capacity == rejected_before[0].capacity &&
         rejected_demands[1].capacity == rejected_before[1].capacity &&
         unique_routes == 31u && templates == 37u;
}

} // namespace node_accel_contract
