#include "../mode.hpp"
#include "../generated_indirect/map.hpp"
#include "internal.hpp"

namespace rund::node::accel::detail::vulkan_persistent_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool build_role(
    VulkanPipeline &pipeline, const std::span<const std::uint32_t> locals,
    const PersistentResidencySlidingRole &source, const std::uint32_t stride,
    const std::uint64_t plan_identity, const std::uint64_t token,
    const std::uint64_t run_generation, VulkanResidencyPersistentRole &role,
    VulkanResidencyPersistentRun *const expected_run,
    const bool tail_role = false) noexcept {
  if (pipeline.residency == nullptr) {
    return false;
  }
  const auto access =
      vulkan_generated_indirect_detail::map_access(*pipeline.residency);
  if (!access.usable()) {
    return false;
  }
  if (access.selected) {
    std::size_t generated_count = 0u;
    return vulkan_sliding_detail::record_generated_map_role(
               pipeline, *pipeline.residency, source, stride, plan_identity,
               token, run_generation, locals,
               std::span<VkCommandBuffer>{role.commands.data(),
                                          role.commands.size()},
               generated_count, role.dispatch_count, role.control_count,
               role.reset_count, role.reset_bytes, expected_run, tail_role) &&
           (role.command_count = generated_count) != 0u;
  }
  std::size_t base_count = 0u;
  const rund::AccelCheck built = BuildVulkanResidencySubmission(
      pipeline, locals, role.commands, base_count, role.dispatch_count,
      role.control_count, role.reset_count, role.reset_bytes);
  if (!built.ok || base_count < 2u || base_count + 1u > role.commands.size()) {
    return false;
  }
  std::move_backward(role.commands.begin() + 1u,
                     role.commands.begin() + base_count,
                     role.commands.begin() + base_count + 1u);
  role.commands[1u] = pipeline.residency->sliding.command.buffer;
  role.command_count = base_count + 1u;
  return role.commands[1u] != VK_NULL_HANDLE;
}

} // namespace

bool materialize(
    const PersistentResidencySlidingRequest &request,
    std::array<VulkanResidencyPersistentRole, ResidencySlidingCapacity> &roles,
    VulkanResidencyPersistentRole &tail, VulkanAdapter *&adapter,
    VulkanResidencyPersistentRun *&run,
    VulkanResidencyPersistentRun *const expected_run) noexcept {
  adapter = nullptr;
  run = expected_run;
  if (request.width != 2u && request.width != 4u) {
    return false;
  }
  for (std::size_t slot = 0u; slot < request.width; ++slot) {
    const PersistentResidencySlidingRole &source = request.roles[slot];
    auto *const pipeline = static_cast<VulkanPipeline *>(source.prepared.get());
    VulkanResidencySelection *const selection =
        pipeline == nullptr ? nullptr : pipeline->residency.get();
    vulkan_generated_indirect_detail::MapAccess access{};
    if (selection != nullptr) {
      access = vulkan_generated_indirect_detail::map_access(*selection);
    }
    if (!ValidVulkanPipeline(pipeline) || selection == nullptr ||
        !access.usable() ||
        !selection->ready || (!selection->bounded && !access.selected) ||
        (!access.selected && !selection->sliding.ready_for_submit) ||
        selection->sliding.quarantined.load(std::memory_order_acquire) ||
        selection->active_window.load(std::memory_order_acquire) != nullptr ||
        selection->active_schedule.load(std::memory_order_acquire) != nullptr ||
        selection->active_persistent.load(std::memory_order_acquire) !=
            expected_run ||
        (adapter != nullptr && adapter != pipeline->adapter)) {
      return false;
    }
    for (std::size_t prior = 0u; prior < slot; ++prior) {
      if (roles[prior].pipeline == pipeline) {
        return false;
      }
    }
    adapter = pipeline->adapter;
    roles[slot].pipeline = pipeline;
    if (!build_role(*pipeline,
                    std::span<const std::uint32_t>{source.locals.data(),
                                                   source.local_count},
                    source, request.width, request.plan_identity, request.token,
                    request.generation, roles[slot], expected_run, false)) {
      return false;
    }
    if (slot == 0u) {
      run = &selection->persistent;
    }
  }
  const std::size_t tail_slot =
      static_cast<std::size_t>((request.coordinate_count - 1u) % request.width);
  const PersistentResidencySlidingRole &source = request.roles[tail_slot];
  tail.pipeline = roles[tail_slot].pipeline;
  if (request.tail_local_count == source.local_count) {
    tail = roles[tail_slot];
  } else if (!build_role(*tail.pipeline,
                         std::span<const std::uint32_t>{
                             source.locals.data(), request.tail_local_count},
                         source, request.width, request.plan_identity,
                         request.token, request.generation, tail, expected_run,
                         true)) {
    return false;
  }
  return adapter != nullptr && run == expected_run && run != nullptr;
}

#endif

} // namespace rund::node::accel::detail::vulkan_persistent_detail
