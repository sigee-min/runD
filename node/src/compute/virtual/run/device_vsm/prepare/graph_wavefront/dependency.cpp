#include "../graph_wavefront.hpp"

#include "../../../../../../accel/kernel/residency/device_vsm/graph_wavefront.hpp"

#include <array>
#include <cstddef>
#include <limits>

namespace rund::compute::detail::device_vsm_product_detail {
namespace {

[[nodiscard]] bool
append_dependency(const residency::TiledGraphStageDependency dependency,
                  const std::uint32_t stage_count, std::uint8_t &dispatch,
                  std::uint8_t &release) noexcept {
  if (dependency.stage >= stage_count || dependency.stage >= 8u) {
    return false;
  }
  const auto bit = static_cast<std::uint8_t>(1u << dependency.stage);
  std::uint8_t &target =
      dependency.phase == residency::TiledGraphDependencyPhase::ReleaseComplete
          ? release
          : dispatch;
  if ((dispatch & bit) != 0u || (release & bit) != 0u) {
    return false;
  }
  target = static_cast<std::uint8_t>(target | bit);
  return true;
}

[[nodiscard]] bool
graph_masks_match(const residency::TiledGraphPlan &plan,
                  const node::accel::detail::DeviceVsmGraphWavefrontProof
                      &wavefront) noexcept {
  namespace accel = node::accel::detail;
  std::array<std::uint8_t, accel::DeviceVsmGraphStageCapacity> dispatch{};
  std::array<std::uint8_t, accel::DeviceVsmGraphStageCapacity> release{};
  std::array<std::uint8_t, accel::DeviceVsmGraphStageCapacity> prior_dispatch{};
  std::array<std::uint8_t, accel::DeviceVsmGraphStageCapacity> prior_release{};
  const auto add =
      [&](std::array<std::uint8_t, accel::DeviceVsmGraphStageCapacity> &target,
          std::array<std::uint8_t, accel::DeviceVsmGraphStageCapacity> &other,
          const std::uint32_t stage, const std::uint32_t dependency,
          const bool prior) noexcept {
        if (stage >= wavefront.stage_count || (!prior && dependency >= stage) ||
            dependency >= accel::DeviceVsmGraphStageCapacity) {
          return false;
        }
        const std::uint8_t bit = static_cast<std::uint8_t>(1u << dependency);
        if ((other[stage] & bit) != 0u) {
          return false;
        }
        target[stage] = static_cast<std::uint8_t>(target[stage] | bit);
        return true;
      };
  const auto resources = plan.resources();
  const auto stages = plan.stages();
  for (std::size_t stage_index = 0u; stage_index < stages.size();
       ++stage_index) {
    for (const residency::TiledGraphPort port : stages[stage_index].ports) {
      const residency::TiledGraphResource *const current =
          plan.resource(port.resource);
      if (current == nullptr) {
        return false;
      }
      if (port.access == residency::Access::Read &&
          current->persistence == residency::ResourcePersistence::Transient &&
          !add(dispatch, release, static_cast<std::uint32_t>(stage_index),
               current->producer_stage, false)) {
        return false;
      }
      if (current->first_stage != stage_index) {
        continue;
      }
      const residency::TiledGraphResource *previous = nullptr;
      const residency::TiledGraphResource *last = current;
      bool first = true;
      for (const residency::TiledGraphResource &candidate : resources) {
        if (candidate.physical_id != current->physical_id) {
          continue;
        }
        if (candidate.first_stage < current->first_stage) {
          first = false;
        }
        if (candidate.last_stage < current->first_stage &&
            (previous == nullptr ||
             previous->last_stage < candidate.last_stage)) {
          previous = &candidate;
        }
        if (last->last_stage < candidate.last_stage) {
          last = &candidate;
        }
      }
      if (previous != nullptr &&
          !add(dispatch, release, static_cast<std::uint32_t>(stage_index),
               previous->last_stage, false)) {
        return false;
      }
      if (first && !add(prior_dispatch, prior_release,
                        static_cast<std::uint32_t>(stage_index),
                        last->last_stage, true)) {
        return false;
      }
      if (previous != nullptr &&
          previous->role == residency::GraphResourceRole::Output) {
        release[stage_index] = static_cast<std::uint8_t>(
            release[stage_index] | (std::uint8_t{1u} << previous->last_stage));
        dispatch[stage_index] = static_cast<std::uint8_t>(
            dispatch[stage_index] &
            ~(std::uint8_t{1u} << previous->last_stage));
      }
      if (first && last->role == residency::GraphResourceRole::Output) {
        prior_release[stage_index] =
            static_cast<std::uint8_t>(prior_release[stage_index] |
                                      (std::uint8_t{1u} << last->last_stage));
        prior_dispatch[stage_index] =
            static_cast<std::uint8_t>(prior_dispatch[stage_index] &
                                      ~(std::uint8_t{1u} << last->last_stage));
      }
    }
  }
  for (std::size_t stage = 0u; stage < stages.size(); ++stage) {
    if (dispatch[stage] != wavefront.same_dispatch[stage] ||
        release[stage] != wavefront.same_release[stage] ||
        prior_dispatch[stage] != wavefront.prior_dispatch[stage] ||
        prior_release[stage] != wavefront.prior_release[stage]) {
      return false;
    }
  }
  return true;
}

} // namespace

bool project_graph_wavefront(
    const residency::TiledGraphPlan &plan,
    const std::uint64_t active_page_count, const std::uint32_t map_stage,
    const std::uint32_t collective_stage,
    node::accel::detail::DeviceVsmGraphWavefrontProof &result) noexcept {
  namespace accel = node::accel::detail;
  result = {};
  const std::span<const residency::TiledGraphStage> stages = plan.stages();
  if (active_page_count == 0u ||
      active_page_count > std::numeric_limits<std::uint32_t>::max() ||
      plan.frame_capacity() == 0u ||
      plan.frame_capacity() > active_page_count || stages.size() < 2u ||
      stages.size() > accel::DeviceVsmGraphStageCapacity ||
      map_stage >= stages.size() || collective_stage >= stages.size() ||
      map_stage == collective_stage) {
    return false;
  }
  const std::uint64_t batches =
      active_page_count / plan.frame_capacity() +
      static_cast<std::uint64_t>(active_page_count % plan.frame_capacity() !=
                                 0u);
  if (plan.frame_capacity() > std::numeric_limits<std::uint32_t>::max() ||
      batches > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  result.stage_count = static_cast<std::uint32_t>(stages.size());
  result.frame_capacity = static_cast<std::uint32_t>(plan.frame_capacity());
  result.batch_count = static_cast<std::uint32_t>(batches);
  result.map_stage = map_stage;
  result.collective_stage = collective_stage;
  for (std::size_t stage = 0u; stage < stages.size(); ++stage) {
    std::uint8_t same_dispatch = 0u;
    std::uint8_t same_release = 0u;
    for (const residency::TiledGraphStageDependency dependency :
         stages[stage].same_batch_predecessors) {
      if (!append_dependency(dependency, result.stage_count, same_dispatch,
                             same_release)) {
        return false;
      }
    }
    result.same_dispatch[stage] = same_dispatch;
    result.same_release[stage] = same_release;
    for (const residency::TiledGraphStageDependency dependency :
         stages[stage].prior_batch_predecessors) {
      if (!append_dependency(dependency, result.stage_count,
                             result.prior_dispatch[stage],
                             result.prior_release[stage])) {
        return false;
      }
    }
  }
  return graph_masks_match(plan, result) &&
         accel::device_vsm_graph_wavefront_valid(result, active_page_count);
}

} // namespace rund::compute::detail::device_vsm_product_detail
