#include "internal.hpp"

#include "../generated_indirect/internal.hpp"
#include "../generated_indirect/map.hpp"

namespace rund::node::accel::detail::vulkan_persistent_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

bool control_generation(const PersistentResidencySlidingRole &role,
                        const std::uint64_t turn,
                        std::uint32_t &generation) noexcept {
  if (turn > (std::numeric_limits<std::uint32_t>::max() -
              role.first_control_generation) /
                 role.control_generation_stride) {
    generation = 0u;
    return false;
  }
  generation = static_cast<std::uint32_t>(
      role.first_control_generation + turn * role.control_generation_stride);
  return generation != 0u;
}

bool descriptor_generation(const PersistentResidencySlidingRole &role,
                           const std::uint64_t turn,
                           std::uint64_t &generation) noexcept {
  if (turn > (std::numeric_limits<std::uint64_t>::max() -
              role.first_descriptor_generation) /
                 role.descriptor_generation_stride) {
    generation = 0u;
    return false;
  }
  generation = role.first_descriptor_generation +
               turn * role.descriptor_generation_stride;
  return generation != 0u;
}

const PersistentResidencySlidingRole &
source_role(const VulkanResidencyPersistentRun &run,
            const std::uint64_t coordinate) noexcept {
  return run.request
      .roles[static_cast<std::size_t>(coordinate % run.request.width)];
}

VulkanResidencyPersistentRole &
native_role(VulkanResidencyPersistentRun &run,
            const std::uint64_t coordinate) noexcept {
  return coordinate + 1u == run.request.coordinate_count
             ? run.tail
             : run.roles[static_cast<std::size_t>(coordinate %
                                                  run.request.width)];
}

std::size_t local_count(const VulkanResidencyPersistentRun &run,
                        const std::uint64_t coordinate) noexcept {
  return coordinate + 1u == run.request.coordinate_count
             ? run.request.tail_local_count
             : source_role(run, coordinate).local_count;
}

VulkanTimelinePoint point(const VulkanResidencyPersistentRun &run,
                          const std::uint64_t coordinate) noexcept {
  const std::size_t cell =
      static_cast<std::size_t>(coordinate % VulkanResidencyWindowCapacity);
  const std::uint64_t turn = coordinate / VulkanResidencyWindowCapacity;
  return VulkanTimelinePoint{.generation = run.timeline_generation,
                             .value = run.done_base + coordinate,
                             .ready_value = run.ready_base[cell] + turn,
                             .ready_cell = static_cast<std::uint8_t>(cell)};
}

void clear_active(VulkanResidencyPersistentRun &run) noexcept {
  for (std::size_t slot = 0u; slot < run.request.width; ++slot) {
    auto *const pipeline =
        static_cast<VulkanPipeline *>(run.request.roles[slot].prepared.get());
    if (pipeline == nullptr || pipeline->residency == nullptr) {
      continue;
    }
    VulkanResidencyPersistentRun *expected = &run;
    static_cast<void>(
        pipeline->residency->active_persistent.compare_exchange_strong(
            expected, nullptr, std::memory_order_acq_rel,
            std::memory_order_acquire));
  }
}

bool gate_result(VulkanResidencySelection &selection,
                 const std::uint64_t descriptor, bool &known_failure,
                 const char *&reason) noexcept {
  known_failure = false;
  reason = "accel_kernel_pipeline_invalid";
  const auto access =
      vulkan_generated_indirect_detail::map_access(selection);
  if (!access.usable()) {
    return false;
  }
  if (access.selected) {
    return vulkan_generated_indirect_detail::gate_result(
        selection.sliding, descriptor, known_failure, reason);
  }
  VulkanResidencySlidingGate &gate = selection.sliding;
  using namespace vulkan_sliding_detail;
  if (gate.descriptor.mapped == nullptr ||
      gate.descriptor.bytes < sizeof(VulkanResidencySlidingPayload)) {
    return false;
  }
  gate.gpu_result_read_count.fetch_add(1u, std::memory_order_relaxed);
  const auto *const payload =
      static_cast<const VulkanResidencySlidingPayload *>(
          gate.descriptor.mapped);
  const std::uint64_t observed =
      static_cast<std::uint64_t>(
          payload->words[VulkanResidencySlidingObservedGenerationWord]) |
      (static_cast<std::uint64_t>(
           payload->words[VulkanResidencySlidingObservedGenerationWord + 1u])
       << 32u);
  if (payload->words[VulkanResidencySlidingAcceptedWord] == 1u &&
      payload->words[VulkanResidencySlidingReasonWord] == 0u &&
      observed == descriptor) {
    return true;
  }
  return false;
}

#endif

} // namespace rund::node::accel::detail::vulkan_persistent_detail
