#include "../../../../../kernel/prepared/interface/api.hpp"
#include "../../../../../kernel/prepared/model.hpp"

#include "../generated_indirect/map.hpp"
#include "../mode.hpp"
#include "internal.hpp"

#include <array>
#include <limits>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] bool
valid_mode(const PersistentResidencySlidingMode mode) noexcept {
  return mode == PersistentResidencySlidingMode::OneSubmit ||
         mode == PersistentResidencySlidingMode::BackendChunked;
}

[[nodiscard]] PersistentResidencySlidingCapability
invalid_capability(const ResidencySlidingMemory memory,
                   const PersistentResidencySlidingMode mode) noexcept {
  return PersistentResidencySlidingCapability{
      .check = {false, "accel_kernel_pipeline_invalid"},
      .memory = memory,
      .mode = mode,
  };
}

[[nodiscard]] PersistentResidencySlidingCapability
unsupported_capability(const ResidencySlidingMemory memory,
                       const PersistentResidencySlidingMode mode) noexcept {
  return PersistentResidencySlidingCapability{
      .check = {false, "compute_backend_unsupported"},
      .memory = memory,
      .mode = mode,
  };
}

} // namespace

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace vulkan_persistent_detail {

PersistentResidencySlidingCapability
capability_for(const std::span<const PersistentResidencySlidingRole> roles,
               const std::uint64_t coordinate_count,
               const ResidencySlidingMemory memory,
               const PersistentResidencySlidingMode mode) noexcept {
  if (roles.empty() || roles.size() > PersistentResidencySlidingCapacity ||
      coordinate_count == 0u ||
      memory != ResidencySlidingMemory::HostCoherent || !valid_mode(mode)) {
    return invalid_capability(memory, mode);
  }
  if (roles.size() != 2u && roles.size() != 4u) {
    return unsupported_capability(memory, mode);
  }
  VulkanAdapter *adapter = nullptr;
  VulkanPipeline *representative = nullptr;
  for (std::size_t slot = 0u; slot < roles.size(); ++slot) {
    const PersistentResidencySlidingRole &role = roles[slot];
    auto *const pipeline = static_cast<VulkanPipeline *>(role.prepared.get());
    vulkan_generated_indirect_detail::MapAccess access{};
    if (pipeline != nullptr && pipeline->residency != nullptr) {
      access =
          vulkan_generated_indirect_detail::map_access(*pipeline->residency);
    }
    if (representative != nullptr && pipeline != nullptr &&
        pipeline->residency != nullptr &&
        pipeline->residency->mode != representative->residency->mode) {
      return invalid_capability(memory, mode);
    }
    if (role.slot != slot || role.local_count == 0u ||
        role.local_count > ResidencyWindowLocalCapacity ||
        role.first_control_generation == 0u ||
        role.control_generation_stride == 0u ||
        role.first_descriptor_generation == 0u ||
        role.descriptor_generation_stride == 0u ||
        !ValidVulkanPipeline(pipeline) || pipeline->residency == nullptr ||
        !pipeline->residency->ready ||
        (!pipeline->residency->bounded && !access.selected) ||
        !access.usable() ||
        (!access.selected && !pipeline->residency->sliding.ready_for_submit) ||
        pipeline->residency->sliding.quarantined.load(
            std::memory_order_acquire) ||
        pipeline->adapter->residency_quarantined.load(
            std::memory_order_acquire) ||
        (adapter != nullptr && adapter != pipeline->adapter)) {
      return invalid_capability(memory, mode);
    }
    for (std::size_t prior = 0u; prior < slot; ++prior) {
      if (roles[prior].prepared.get() == role.prepared.get()) {
        return invalid_capability(memory, mode);
      }
    }
    adapter = pipeline->adapter;
    if (representative == nullptr) {
      representative = pipeline;
    }
  }
  if (adapter == nullptr ||
      coordinate_count > std::numeric_limits<std::uint32_t>::max() ||
      coordinate_count > adapter->persistent_stream_submit_capacity) {
    return PersistentResidencySlidingCapability{
        .check = {false, "compute_pipeline_capacity"},
        .memory = memory,
        .mode = mode};
  }
  constexpr std::uint64_t PerBatch =
      sizeof(VulkanTimelineBatch) + sizeof(std::uint64_t) * 2u +
      sizeof(VkPipelineStageFlags) + sizeof(VkTimelineSemaphoreSubmitInfoKHR) +
      sizeof(VkSubmitInfo);
  if (PerBatch > std::numeric_limits<std::uint64_t>::max() /
                     vulkan_persistent_detail::BatchCount) {
    return PersistentResidencySlidingCapability{
        .check = {false, "compute_pipeline_capacity"},
        .memory = memory,
        .mode = mode};
  }
  return PersistentResidencySlidingCapability{
      .check = {true, "ok"},
      .memory = memory,
      .retained_bytes = ::rund::detail::counter::SaturatingAdd(
          sizeof(vulkan_persistent_detail::Owner),
          representative->residency->sliding.retained_bytes),
      .transient_bytes = PerBatch * vulkan_persistent_detail::BatchCount,
      .width = static_cast<std::uint8_t>(roles.size()),
      .whole_run_preencoded = true,
      .device_generated_recurrence = false,
      .fixed_native_storage = false,
      .fixed_common_storage = false,
      .host_epoch_callbacks_zero = true,
      .mode = mode,
  };
}

} // namespace vulkan_persistent_detail

PersistentResidencySlidingCapability VulkanResidencyPersistentCapability(
    const std::span<const PersistentResidencySlidingRole> roles,
    const std::uint64_t coordinate_count,
    const ResidencySlidingMemory memory) noexcept {
  return vulkan_persistent_detail::capability_for(
      roles, coordinate_count, memory,
      PersistentResidencySlidingMode::OneSubmit);
}

#endif

PersistentResidencySlidingCapability
QueryVulkanPreparedPersistentSlidingCapability(
    const std::span<const PreparedResidencyPersistentSlidingRole> roles,
    const std::uint64_t coordinate_count, const ResidencySlidingMemory memory,
    const PersistentResidencySlidingMode mode) noexcept {
  if (roles.empty() || roles.size() > PersistentResidencySlidingCapacity ||
      coordinate_count == 0u ||
      memory != ResidencySlidingMemory::HostCoherent || !valid_mode(mode)) {
    return invalid_capability(memory, mode);
  }

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  std::array<PersistentResidencySlidingRole, PersistentResidencySlidingCapacity>
      native{};
  const prepared::PipelineState *first = nullptr;
  for (std::size_t slot = 0u; slot < roles.size(); ++slot) {
    const PreparedResidencyPersistentSlidingRole &role = roles[slot];
    if (!role.pipeline.ok || role.pipeline.owner == nullptr ||
        role.slot != slot || role.local_count == 0u ||
        role.local_count > ResidencyWindowLocalCapacity ||
        role.first_control_generation == 0u ||
        role.control_generation_stride == 0u ||
        role.first_descriptor_generation == 0u ||
        role.descriptor_generation_stride == 0u) {
      return invalid_capability(memory, mode);
    }
    const auto *const state =
        static_cast<const prepared::PipelineState *>(role.pipeline.owner.get());
    if (state == nullptr || state->ops == nullptr ||
        state->backend == nullptr ||
        state->ops->api != rund::AccelApi::Vulkan) {
      return invalid_capability(memory, mode);
    }
    if (first == nullptr) {
      first = state;
    } else if (state->ops != first->ops) {
      return invalid_capability(memory, mode);
    }
    for (std::size_t prior = 0u; prior < slot; ++prior) {
      if (roles[prior].pipeline.owner.get() == role.pipeline.owner.get()) {
        return invalid_capability(memory, mode);
      }
    }
    native[slot] = PersistentResidencySlidingRole{
        .prepared = state->backend,
        .locals = role.locals,
        .local_count = role.local_count,
        .first_control_generation = role.first_control_generation,
        .control_generation_stride = role.control_generation_stride,
        .first_descriptor_generation = role.first_descriptor_generation,
        .descriptor_generation_stride = role.descriptor_generation_stride,
        .slot = role.slot,
    };
  }
  if (roles.size() != 2u && roles.size() != 4u) {
    return unsupported_capability(memory, mode);
  }
  if (first == nullptr || first->ops->prepare_persistent_sliding == nullptr) {
    return unsupported_capability(memory, mode);
  }
  return vulkan_persistent_detail::capability_for(
      std::span<const PersistentResidencySlidingRole>{native.data(),
                                                      roles.size()},
      coordinate_count, memory, mode);
#else
  (void)roles;
  return unsupported_capability(memory, mode);
#endif
}

} // namespace rund::node::accel::detail
