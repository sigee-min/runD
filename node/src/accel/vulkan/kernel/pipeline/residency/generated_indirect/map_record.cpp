#include "internal.hpp"
#include "map.hpp"
#include "map_state.hpp"

#include "../../../../command/resources.hpp"
#include "../../../../map/api.hpp"
#include "../../../../map/local.hpp"
#include "../../prepare/record.hpp"

#include <algorithm>
#include <array>
#include <kernel/core/checked.hpp>

namespace rund::node::accel::detail::vulkan_generated_indirect_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

bool record_role(VulkanPipeline &pipeline, VulkanResidencySelection &selection,
                 const PersistentResidencySlidingRole &source,
                 const std::uint32_t stride, const std::uint64_t plan_identity,
                 const std::uint64_t token, const std::uint64_t run_generation,
                 const std::span<const std::uint32_t> locals,
                 const std::span<VkCommandBuffer> commands,
                 std::size_t &command_count, std::uint64_t &dispatch_count,
                 std::uint64_t &control_count, std::uint64_t &reset_count,
                 std::uint64_t &reset_bytes,
                 VulkanResidencyPersistentRun *const expected_run,
                 const bool tail) noexcept {
  const MapAccess access = map_access(selection);
  if (!access.usable() || !access.selected ||
      pipeline.record == nullptr || selection.adapter != pipeline.adapter ||
      source.slot >= ResidencySlidingCapacity || stride == 0u ||
      source.slot >= stride || source.local_count == 0u ||
      source.local_count > ResidencyWindowLocalCapacity || locals.empty() ||
      locals.size() > source.local_count ||
      (!tail && locals.size() != source.local_count) ||
      locals.size() > selection.steps.size() ||
      commands.size() < locals.size() + 2u ||
      commands.size() > VulkanResidencyCommandCapacity ||
      expected_run == nullptr ||
      selection.active_persistent.load(std::memory_order_acquire) !=
          expected_run) {
    return false;
  }
  std::array<VkCommandBuffer, VulkanResidencyCommandCapacity> old_commands{};
  std::copy(commands.begin(), commands.end(), old_commands.begin());
  const std::size_t old_command_count = command_count;
  const std::uint64_t old_dispatch_count = dispatch_count;
  const std::uint64_t old_control_count = control_count;
  const std::uint64_t old_reset_count = reset_count;
  const std::uint64_t old_reset_bytes = reset_bytes;
  std::array<VulkanMapEncodeResources *, PreparedPipelineStepCapacity> maps{};
  std::array<VulkanMapEncodeResources *, PreparedPipelineStepCapacity>
      saved_maps{};
  std::array<GeneratedMapState, PreparedPipelineStepCapacity> saved_states{};
  std::array<bool, PreparedPipelineStepCapacity> saved{};
  std::size_t map_count = 0u;
  for (std::size_t index = 0u; index < pipeline.record->entries.size();
       ++index) {
    const VulkanPipelineRecordEntry &record_entry =
        pipeline.record->entries[index];
    auto *const resources =
        static_cast<VulkanKernelResources *>(record_entry.prepared.get());
    if (resources == nullptr || resources->size() != 1u ||
        map_count == maps.size()) {
      return false;
    }
    const VulkanKernelEntry *const step = resources->entry(0u);
    if (step == nullptr || step->ops.encode != EncodeVulkanMap) {
      return false;
    }
    auto *const map =
        static_cast<VulkanMapEncodeResources *>(step->resource.get());
    if (map == nullptr || map->prepared == nullptr) {
      return false;
    }
    maps[map_count++] = map;
    saved_maps[index] = map;
    saved_states[index] = save_map_state(*map);
    saved[index] = true;
  }
  VulkanMapEncodeResources *controlled = nullptr;
  for (std::size_t index = 0u; index < map_count; ++index) {
    VulkanMapEncodeResources *const map = maps[index];
    if (!map->prepared->checks.empty()) {
      if (controlled != nullptr || map->generated_control_pipeline == nullptr ||
          map->generated_control_descriptor == VK_NULL_HANDLE ||
          map->generated_gate_binding.buffer == VK_NULL_HANDLE ||
          map->generated_summary_binding.buffer == VK_NULL_HANDLE) {
        return false;
      }
      controlled = map;
    }
  }
  if (controlled == nullptr) {
    return false;
  }
  std::array<bool, PreparedPipelineStepCapacity> seen{};
  std::array<VulkanCommand, ResidencyWindowLocalCapacity> temporary{};
  std::array<bool, ResidencyWindowLocalCapacity> made{};
  std::size_t next_command_count = 0u;
  std::uint64_t next_dispatch_count = 0u;
  std::uint64_t next_control_count = 0u;
  std::uint64_t next_reset_count = 0u;
  std::uint64_t next_reset_bytes = 0u;
  const auto restore_maps = [&]() noexcept {
    for (std::size_t index = 0u; index < saved_maps.size(); ++index) {
      if (saved[index] && saved_maps[index] != nullptr) {
        restore_map_state(*saved_maps[index], saved_states[index]);
      }
    }
  };
  const auto restore = [&]() noexcept {
    for (std::size_t index = 0u; index < temporary.size(); ++index) {
      if (made[index]) {
        DestroyCommand(pipeline.adapter->device, temporary[index]);
      }
    }
    restore_maps();
    std::copy(old_commands.begin(), old_commands.begin() + commands.size(),
              commands.begin());
    command_count = old_command_count;
    dispatch_count = old_dispatch_count;
    control_count = old_control_count;
    reset_count = old_reset_count;
    reset_bytes = old_reset_bytes;
  };
  if (selection.prefix.buffer == VK_NULL_HANDLE ||
      selection.suffix.buffer == VK_NULL_HANDLE) {
    restore();
    return false;
  }
  for (std::size_t index = 0u; index < locals.size(); ++index) {
    const std::uint32_t local = locals[index];
    if (local >= seen.size() || seen[local] || local >= source.local_count ||
        source.locals[index] != local || local >= selection.steps.size() ||
        selection.steps[local].declared_step != local ||
        selection.steps[local].dispatch_count == 0u) {
      restore();
      return false;
    }
    seen[local] = true;
  }
  if (locals.size() > temporary.size()) {
    restore();
    return false;
  }
  {
    MapModeGuard generated{*controlled};
    controlled->generated_owner = static_cast<std::uint64_t>(
        reinterpret_cast<std::uintptr_t>(&pipeline));
    controlled->generated_plan = plan_identity;
    controlled->generated_token = token;
    controlled->generated_run = run_generation;
    controlled->generated_slot = source.slot;
    controlled->generated_stride = stride;
    controlled->generated_first_control_generation =
        source.first_control_generation;
    controlled->generated_control_generation_stride =
        source.control_generation_stride;
    controlled->generated_first_descriptor_generation =
        source.first_descriptor_generation;
    controlled->generated_descriptor_generation_stride =
        source.descriptor_generation_stride;
    for (std::size_t index = 0u; index < locals.size(); ++index) {
      const std::uint32_t local = locals[index];
      VulkanCommand &command = temporary[index];
      const rund::AccelCheck created = CreateCommand(
          pipeline.adapter->device, pipeline.adapter->compute_queue_family,
          command, CommandKind::ReusablePrimary);
      if (!created.ok) {
        restore();
        return false;
      }
      made[index] = true;
      const VulkanPipelineRecordSlice slice{.first = local, .count = 1u};
      const rund::AccelCheck recorded =
          RecordVulkanPipeline(pipeline, command, CommandKind::ReusablePrimary,
                               false, nullptr, nullptr, &slice);
      if (!recorded.ok) {
        restore();
        return false;
      }
      if (!rund::kernel::checked::add(next_dispatch_count,
                                      selection.steps[local].dispatch_count,
                                      next_dispatch_count) ||
          !rund::kernel::checked::add(next_control_count,
                                      selection.steps[local].control_count,
                                      next_control_count) ||
          !rund::kernel::checked::add(next_reset_count,
                                      selection.steps[local].reset_count,
                                      next_reset_count) ||
          !rund::kernel::checked::add(next_reset_bytes,
                                      selection.steps[local].reset_bytes,
                                      next_reset_bytes)) {
        restore();
        return false;
      }
    }
  }
  restore_maps();
  auto &owned = tail ? selection.sliding.generated_tail
                     : selection.sliding.generated_roles[source.slot];
  for (VulkanCommand &command : owned) {
    DestroyCommand(pipeline.adapter->device, command);
  }
  for (std::size_t index = 0u; index < owned.size(); ++index) {
    owned[index] = {};
  }
  for (std::size_t index = 0u; index < locals.size(); ++index) {
    const std::uint32_t local = locals[index];
    owned[local] = temporary[index];
    temporary[index] = {};
    made[index] = false;
  }
  next_command_count = 1u + locals.size() + 1u;
  commands[0u] = selection.prefix.buffer;
  for (std::size_t index = 0u; index < locals.size(); ++index) {
    commands[index + 1u] = owned[locals[index]].buffer;
  }
  commands[next_command_count - 1u] = selection.suffix.buffer;
  command_count = next_command_count;
  dispatch_count = next_dispatch_count;
  control_count = next_control_count;
  reset_count = next_reset_count;
  reset_bytes = next_reset_bytes;
  return true;
}

#endif

} // namespace rund::node::accel::detail::vulkan_generated_indirect_detail
