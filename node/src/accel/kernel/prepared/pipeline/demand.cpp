#include "../run.hpp"
#include "demand.hpp"

#include <array>
#include <limits>

namespace rund::node::accel::detail {

[[nodiscard]] bool
backend_template_route_demand(const std::uint64_t owner_count,
                              const std::uint64_t route_copies,
                              BackendTemplateRouteDemand &demand) noexcept {
  std::uint64_t capacity = 0u;
  if (owner_count == 0u ||
      owner_count > std::numeric_limits<std::uint32_t>::max() ||
      (route_copies != 1u && route_copies != 2u) ||
      !rund::kernel::checked::mul(owner_count, route_copies, capacity) ||
      capacity > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  const BackendTemplateRouteDemand candidate{
      .owner_count = static_cast<std::uint32_t>(owner_count),
      .route_copies = static_cast<std::uint32_t>(route_copies),
      .capacity = static_cast<std::uint32_t>(capacity),
  };
  if (!candidate.valid()) {
    return false;
  }
  demand = candidate;
  return true;
}

// Freezes one exact demand for every route entry before the first native
// materialization. Duplicate entries that borrow one RunState do not increase
// demand. Structurally equal but independently bound RunStates do. Equality is
// checked in both directions and must form one complete equivalence class;
// an asymmetric or non-transitive backend predicate fails closed.
[[nodiscard]] bool plan_backend_template_route_demands(
    const std::span<const PreparedKernelRun *const> runs,
    const std::uint32_t route_copies,
    const std::span<BackendTemplateRouteDemand> demands,
    std::uint64_t &unique_route_count, std::uint64_t &template_count) noexcept {
  if (runs.empty() || runs.size() > PreparedPipelineStepCapacity ||
      demands.size() != runs.size() ||
      (route_copies != 1u && route_copies != 2u)) {
    return false;
  }

  std::array<std::size_t, PreparedPipelineStepCapacity> unique_entries{};
  std::array<std::size_t, PreparedPipelineStepCapacity> entry_unique{};
  std::array<std::size_t, PreparedPipelineStepCapacity> unique_group{};
  std::array<std::uint64_t, PreparedPipelineStepCapacity> group_counts{};
  std::array<BackendTemplateRouteDemand, PreparedPipelineStepCapacity>
      candidates{};
  const BackendOps *common_ops = nullptr;
  std::size_t unique_count = 0u;

  for (std::size_t index = 0u; index < runs.size(); ++index) {
    const PreparedKernelRun *const item = runs[index];
    const auto *const state =
        item == nullptr
            ? nullptr
            : static_cast<const prepared::RunState *>(item->owner.get());
    const BackendOps *const ops =
        state == nullptr ? nullptr : state->bound.run.ops;
    if (item == nullptr || !item->ok || state == nullptr || ops == nullptr ||
        ops->same_pipeline_template == nullptr ||
        !IsPipelinePrivatePreparation(state->mode) ||
        (common_ops != nullptr && common_ops != ops)) {
      return false;
    }
    common_ops = ops;

    std::size_t duplicate = unique_count;
    for (std::size_t prior = 0u; prior < unique_count; ++prior) {
      const PreparedKernelRun *const previous = runs[unique_entries[prior]];
      if (previous != nullptr && previous->owner.get() == item->owner.get()) {
        duplicate = prior;
        break;
      }
    }
    if (duplicate != unique_count) {
      entry_unique[index] = duplicate;
      continue;
    }

    std::size_t matched_group = unique_count;
    std::uint64_t matched_members = 0u;
    const BackendRun &run = state->bound.run;
    if (!ops->same_pipeline_template(run, run)) {
      return false;
    }
    for (std::size_t prior = 0u; prior < unique_count; ++prior) {
      const PreparedKernelRun *const previous = runs[unique_entries[prior]];
      const auto *const previous_state =
          previous == nullptr
              ? nullptr
              : static_cast<const prepared::RunState *>(previous->owner.get());
      if (previous_state == nullptr) {
        return false;
      }
      const BackendRun &prior_run = previous_state->bound.run;
      const bool forward = ops->same_pipeline_template(run, prior_run);
      const bool reverse = ops->same_pipeline_template(prior_run, run);
      if (forward != reverse) {
        return false;
      }
      if (!forward) {
        continue;
      }
      ++matched_members;
      const std::size_t prior_group = unique_group[prior];
      if (matched_group == unique_count) {
        matched_group = prior_group;
      } else if (matched_group != prior_group) {
        return false;
      }
    }
    if (matched_group == unique_count) {
      matched_group = unique_count;
    } else if (matched_members != group_counts[matched_group]) {
      return false;
    }
    unique_entries[unique_count] = index;
    entry_unique[index] = unique_count;
    unique_group[unique_count] = matched_group;
    ++group_counts[matched_group];
    ++unique_count;
  }

  std::uint64_t groups = 0u;
  for (std::size_t unique = 0u; unique < unique_count; ++unique) {
    const std::size_t group = unique_group[unique];
    if (group >= unique_count || group_counts[group] == 0u ||
        !backend_template_route_demand(group_counts[group], route_copies,
                                       candidates[unique_entries[unique]])) {
      return false;
    }
    groups += static_cast<std::uint64_t>(group == unique);
  }
  for (std::size_t index = 0u; index < runs.size(); ++index) {
    const std::size_t unique = entry_unique[index];
    if (unique >= unique_count) {
      return false;
    }
    candidates[index] = candidates[unique_entries[unique]];
  }
  for (std::size_t index = 0u; index < runs.size(); ++index) {
    demands[index] = candidates[index];
  }
  unique_route_count = unique_count;
  template_count = groups;
  return true;
}

bool BackendTemplateRouteDemandForContract(
    const std::uint64_t owner_count, const std::uint64_t route_copies,
    BackendTemplateRouteDemand &demand) noexcept {
  return backend_template_route_demand(owner_count, route_copies, demand);
}

bool PlanBackendTemplateRouteDemandsForContract(
    const std::span<const PreparedKernelRun *const> runs,
    const std::uint32_t route_copies,
    const std::span<BackendTemplateRouteDemand> demands,
    std::uint64_t &unique_route_count, std::uint64_t &template_count) noexcept {
  return plan_backend_template_route_demands(
      runs, route_copies, demands, unique_route_count, template_count);
}

} // namespace rund::node::accel::detail
