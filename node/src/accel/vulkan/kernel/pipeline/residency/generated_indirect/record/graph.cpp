#include "../internal.hpp"
#include "../map_state.hpp"

#include "../../../../../command/resources.hpp"
#include "../../../../../map/admission.hpp"
#include "../../../../../map/api.hpp"
#include "../../../../../map/local.hpp"
#include "../../../prepare/record.hpp"

#include <array>
#include <kernel/core/checked.hpp>

namespace rund::node::accel::detail::vulkan_generated_indirect_detail::impl {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

bool record_graph(VulkanPipeline &pipeline, VulkanResidencySelection &selection,
                  const std::span<const std::uint32_t> locals) noexcept {
  const auto fail = [&](const char *const reason) noexcept {
    auto &diagnostics = selection.graph_generated_diagnostics;
    diagnostics.failed = true;
    diagnostics.first_reason = reason;
    diagnostics.phase = VulkanResidencyGraphGeneratedPhase::Quarantined;
    diagnostics.quarantined = true;
    selection.graph_generated_phase =
        VulkanResidencyGraphGeneratedPhase::Quarantined;
    selection.quarantined.store(true, std::memory_order_release);
    return false;
  };
  const bool admitted = selection.graph_generated_phase ==
                        VulkanResidencyGraphGeneratedPhase::Admitted;
  if (pipeline.record == nullptr || selection.adapter != pipeline.adapter ||
      selection.mode != VulkanResidencyMode::GraphStageGeneratedIndirect ||
      !selection.graph_generated.valid || locals.empty() ||
      locals.size() != selection.graph_generated.local_count ||
      locals.size() > selection.graph_generated_frames.size() || !admitted) {
    return fail("accel_kernel_pipeline_invalid");
  }
  if (selection.graph_generated_generation == 0u) {
    selection.graph_generated_diagnostics.replay = true;
    return fail("accel_kernel_pipeline_replay");
  }
  std::array<bool, PreparedPipelineStepCapacity> seen{};
  std::array<VulkanCommand, PreparedPipelineStepCapacity> temporary{};
  std::array<bool, PreparedPipelineStepCapacity> made{};
  std::array<GeneratedMapState, PreparedPipelineStepCapacity> saved{};
  std::array<bool, PreparedPipelineStepCapacity> saved_map{};
  const auto expected_generation = [&](const std::uint32_t local,
                                       std::uint64_t &value) noexcept {
    return rund::kernel::checked::add(static_cast<std::uint64_t>(local), 1u,
                                      value);
  };
  const auto cleanup = [&]() noexcept {
    for (std::size_t index = 0u; index < temporary.size(); ++index) {
      if (made[index]) {
        DestroyCommand(pipeline.adapter->device, temporary[index]);
      }
    }
    for (std::size_t index = 0u; index < saved.size(); ++index) {
      if (!saved_map[index] || index >= pipeline.record->entries.size()) {
        continue;
      }
      auto *const resources = static_cast<VulkanKernelResources *>(
          pipeline.record->entries[index].prepared.get());
      auto *const kernel =
          resources == nullptr ? nullptr : resources->entry(0u);
      auto *const map =
          kernel == nullptr
              ? nullptr
              : static_cast<VulkanMapEncodeResources *>(kernel->resource.get());
      if (map != nullptr) {
        restore_map_state(*map, saved[index]);
      }
    }
  };
  for (const std::uint32_t local : locals) {
    if (local >= selection.graph_generated.local_count || seen[local]) {
      cleanup();
      return fail("accel_kernel_pipeline_invalid");
    }
    seen[local] = true;
    if (local >= pipeline.record->entries.size()) {
      cleanup();
      return fail("accel_kernel_pipeline_invalid");
    }
    auto *const resources = static_cast<VulkanKernelResources *>(
        pipeline.record->entries[local].prepared.get());
    auto *const kernel = resources == nullptr ? nullptr : resources->entry(0u);
    auto *const map =
        kernel == nullptr
            ? nullptr
            : static_cast<VulkanMapEncodeResources *>(kernel->resource.get());
    auto &frame = selection.graph_generated_frames[local];
    std::uint64_t expected{};
    if (map == nullptr || map->prepared == nullptr ||
        map->generated_control_pipeline == nullptr ||
        map->generated_control_descriptor == VK_NULL_HANDLE || !frame.ready ||
        !frame.proof.valid || !expected_generation(local, expected) ||
        frame.generation != expected ||
        frame.expected_descriptor_generation != frame.generation) {
      cleanup();
      return fail("accel_kernel_pipeline_invalid");
    }
    if (!map_owner_matches(*map, frame.control_pipeline,
                           frame.control_descriptor, frame.gate,
                           pipeline.control.summary)) {
      cleanup();
      return fail("accel_kernel_pipeline_invalid");
    }
    saved[local] = save_map_state(*map);
    saved_map[local] = true;
  }
  for (const std::uint32_t local : locals) {
    auto *const resources = static_cast<VulkanKernelResources *>(
        pipeline.record->entries[local].prepared.get());
    auto *const kernel = resources == nullptr ? nullptr : resources->entry(0u);
    auto *const map =
        kernel == nullptr
            ? nullptr
            : static_cast<VulkanMapEncodeResources *>(kernel->resource.get());
    auto &frame = selection.graph_generated_frames[local];
    VulkanCommand &command = temporary[local];
    const rund::AccelCheck created_command = CreateCommand(
        pipeline.adapter->device, pipeline.adapter->compute_queue_family,
        command, CommandKind::ReusablePrimary);
    if (!created_command.ok) {
      cleanup();
      return fail(created_command.reason);
    }
    made[local] = true;
    {
      MapModeGuard generated{*map};
      map->generated_owner = static_cast<std::uint64_t>(
          reinterpret_cast<std::uintptr_t>(&pipeline));
      map->generated_plan = frame.proof.op_hash_hi;
      map->generated_token = frame.proof.graph_id_hi;
      map->generated_run = frame.proof.graph_id_lo;
      map->generated_slot = 0u;
      map->generated_stride = 1u;
      map->generated_first_control_generation = local;
      map->generated_control_generation_stride = 0u;
      map->generated_first_descriptor_generation =
          frame.expected_descriptor_generation;
      // A zero descriptor stride marks the per-stage owner. The generated
      // shader still authenticates the descriptor identity but does not bind
      // this Host-stage command to a Persistent generation counter.
      map->generated_descriptor_generation_stride = 0u;
      if (!BindVulkanGeneratedMapAdmission(*map, frame.gate,
                                           pipeline.control.summary) ||
          !map_owner_matches(*map, frame.control_pipeline,
                             frame.control_descriptor, frame.gate,
                             pipeline.control.summary)) {
        cleanup();
        return fail("accel_kernel_pipeline_invalid");
      }
      const VulkanPipelineRecordSlice slice{.first = local, .count = 1u};
      const rund::AccelCheck recorded =
          RecordVulkanPipeline(pipeline, command, CommandKind::ReusablePrimary,
                               false, nullptr, nullptr, &slice);
      if (!recorded.ok) {
        cleanup();
        return fail(recorded.reason);
      }
    }
  }
  for (const std::uint32_t local : locals) {
    DestroyCommand(pipeline.adapter->device,
                   selection.graph_generated_frames[local].command);
    selection.graph_generated_frames[local].command = temporary[local];
    temporary[local] = {};
    made[local] = false;
  }
  selection.graph_generated_command_count = locals.size();
  selection.graph_generated_diagnostics.recorded_count =
      static_cast<std::uint32_t>(locals.size());
  selection.graph_generated_diagnostics.record_generation =
      selection.graph_generated_generation;
  selection.graph_generated_phase =
      VulkanResidencyGraphGeneratedPhase::Recorded;
  selection.graph_generated_diagnostics.phase =
      VulkanResidencyGraphGeneratedPhase::Recorded;
  return true;
}

#endif

} // namespace rund::node::accel::detail::vulkan_generated_indirect_detail::impl
