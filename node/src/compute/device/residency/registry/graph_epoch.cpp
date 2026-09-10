#include "../registry.hpp"
#include "frame.hpp"
#include "internal.hpp"
#include "lease_state.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <span>

namespace rund::compute::detail::residency {
AuthorityResult Authority::begin_graph_transform(
    const std::span<const PageUse> inputs,
    const GraphMaterialization input_materialization,
    const FrameRegion input_region, const std::span<const PageUse> outputs,
    const GraphMaterialization output_materialization,
    const FrameRegion output_region, const std::uint64_t epoch) noexcept {
  if (inputs.empty() || inputs.size() != outputs.size() ||
      inputs.size() > PipelineLeafCapacity ||
      input_materialization.resource == 0u ||
      output_materialization.resource == 0u ||
      input_materialization.resource == output_materialization.resource ||
      input_materialization.key.backing == 0u ||
      output_materialization.key.backing == 0u ||
      input_materialization.key.page != 0u ||
      output_materialization.key.page != 0u ||
      input_materialization.page_bytes == 0u ||
      output_materialization.page_bytes == 0u ||
      input_materialization.page_count == 0u ||
      output_materialization.page_count == 0u) {
    return AuthorityResult{.failure = inputs.size() > PipelineLeafCapacity
                                          ? AuthorityFailure::Capacity
                                          : AuthorityFailure::Invalid};
  }
  std::array<CacheUse, PipelineLeafCapacity> projected_inputs{};
  std::array<CacheUse, PipelineLeafCapacity> projected_outputs{};
  for (std::size_t index = 0u; index < inputs.size(); ++index) {
    if (!project_graph_use(inputs[index], input_materialization, Access::Read,
                           epoch, projected_inputs[index]) ||
        !project_graph_use(outputs[index], output_materialization,
                           Access::Write, epoch, projected_outputs[index])) {
      return AuthorityResult{.failure = AuthorityFailure::Invalid};
    }
  }
  return begin_transform(
      std::span<const CacheUse>{projected_inputs.data(), inputs.size()},
      input_region,
      std::span<const CacheUse>{projected_outputs.data(), outputs.size()},
      output_region);
}

AuthorityResult
Authority::begin_graph(const std::span<const PageUse> uses,
                       const GraphMaterialization materialization,
                       const FrameRegion region,
                       const std::uint64_t epoch) noexcept {
  std::lock_guard lock{gate_};
  return begin_graph_locked(uses, materialization, region, epoch);
}

AuthorityResult
Authority::begin_graph_locked(const std::span<const PageUse> uses,
                              const GraphMaterialization materialization,
                              const FrameRegion region,
                              const std::uint64_t epoch) noexcept {
  if (uses.empty() || uses.size() > PipelineLeafCapacity ||
      materialization.resource == 0u || materialization.key.backing == 0u ||
      materialization.key.page != 0u || materialization.page_bytes == 0u ||
      materialization.page_count == 0u || region.count == 0u ||
      uses.size() > region.count) {
    return AuthorityResult{.failure = uses.size() > PipelineLeafCapacity
                                          ? AuthorityFailure::Capacity
                                          : AuthorityFailure::Invalid};
  }
  const Access expected =
      region.role == FrameRole::Output ? Access::Write : Access::Read;
  std::array<CacheUse, PipelineLeafCapacity> projected{};
  for (std::size_t index = 0u; index < uses.size(); ++index) {
    if (!project_graph_use(uses[index], materialization, expected, epoch,
                           projected[index])) {
      return AuthorityResult{.failure = AuthorityFailure::Invalid};
    }
  }
  AuthorityResult result =
      begin_locked(std::span<const CacheUse>{projected.data(), uses.size()},
                   region.tier, region.role, region.first, region.count, {});
  if (result) {
    const auto slot =
        std::find_if(cycle_state_.epochs.begin(), cycle_state_.epochs.end(),
                     [token = result.lease.token](const LeaseSlot &candidate) {
                       return candidate.token == token;
                     });
    if (slot == cycle_state_.epochs.end()) {
      static_cast<void>(complete_locked(result.lease.token, false, true));
      return AuthorityResult{.failure = AuthorityFailure::Invalid};
    }
    slot->coordinate = epoch;
  }
  return result;
}

} // namespace rund::compute::detail::residency
