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

bool record_sequence(VulkanPipeline &pipeline,
                     VulkanResidencySelection &selection,
                     const std::span<const std::uint32_t> locals) noexcept {
  const auto fail = [&](const char *const reason) noexcept {
    selection.graph_generated_diagnostics.failed = true;
    selection.graph_generated_diagnostics.first_reason = reason;
    selection.graph_generated_diagnostics.phase =
        VulkanResidencyGraphGeneratedPhase::Quarantined;
    selection.graph_generated_diagnostics.quarantined = true;
    selection.graph_generated_phase =
        VulkanResidencyGraphGeneratedPhase::Quarantined;
    selection.quarantined.store(true, std::memory_order_release);
    return false;
  };
  if (pipeline.record == nullptr || selection.adapter != pipeline.adapter ||
      selection.mode != VulkanResidencyMode::GraphStageSequence ||
      !selection.graph_sequence.valid || locals.size() != 2u ||
      selection.graph_sequence_command_count != 0u ||
      selection.graph_generated_phase !=
          VulkanResidencyGraphGeneratedPhase::Admitted) {
    return fail("accel_kernel_pipeline_invalid");
  }
  std::array<bool, PreparedPipelineStepCapacity> seen{};
  std::array<VulkanCommand, PreparedPipelineStepCapacity> temporary{};
  std::array<bool, PreparedPipelineStepCapacity> made{};
  std::array<std::array<VulkanMapEncodeResources *, 2u>, 2u> saved_maps{};
  std::array<std::array<GeneratedMapState, 2u>, 2u> saved_states{};
  std::array<bool, 2u> saved{};
  const auto restore_maps = [&]() noexcept {
    for (std::size_t local = 0u; local < saved.size(); ++local) {
      if (!saved[local]) {
        continue;
      }
      for (std::size_t stage = 0u; stage < saved_maps[local].size(); ++stage) {
        if (saved_maps[local][stage] != nullptr) {
          restore_map_state(*saved_maps[local][stage],
                            saved_states[local][stage]);
        }
      }
    }
  };
  const auto cleanup = [&]() noexcept {
    for (std::size_t index = 0u; index < temporary.size(); ++index) {
      if (made[index]) {
        DestroyCommand(pipeline.adapter->device, temporary[index]);
      }
    }
    restore_maps();
  };
  for (const std::uint32_t local : locals) {
    if (local >= selection.graph_sequence.local_count || seen[local] ||
        selection.graph_sequence_frames[local].command_ready) {
      cleanup();
      return fail("accel_kernel_pipeline_invalid");
    }
    seen[local] = true;
    auto &frame = selection.graph_sequence_frames[local];
    for (std::size_t stage = 0u; stage < 2u; ++stage) {
      if (!frame.ready[stage] || frame.gates[stage].buffer == VK_NULL_HANDLE ||
          frame.control_pipelines[stage] == nullptr ||
          frame.control_descriptors[stage] == VK_NULL_HANDLE) {
        cleanup();
        return fail("accel_kernel_pipeline_invalid");
      }
      std::uint64_t expected{};
      if (!rund::kernel::checked::add(
              static_cast<std::uint64_t>(local * 2u + stage), 1u, expected) ||
          frame.generations[stage] != expected) {
        cleanup();
        return fail("accel_kernel_pipeline_replay");
      }
    }
  }
  for (std::size_t local = 0u; local < 2u; ++local) {
    if (!seen[local]) {
      cleanup();
      return fail("accel_kernel_pipeline_invalid");
    }
    const auto &frame = selection.graph_sequence_frames[local];
    const auto status_range = pipeline.record->status_ranges[local];
    const std::size_t status_end =
        static_cast<std::size_t>(status_range.first) + status_range.count;
    if (status_range.count != 2u ||
        status_end > pipeline.record->status_steps.size() ||
        pipeline.record->status_steps[status_range.first + 1u].count == 0u) {
      cleanup();
      return fail("accel_vulkan_command_unavailable");
    }
    for (std::size_t stage = 0u; stage < 2u; ++stage) {
      auto *const resources = static_cast<VulkanKernelResources *>(
          pipeline.record->entries[local].prepared.get());
      auto *const kernel =
          resources == nullptr ? nullptr : resources->entry(stage);
      auto *const map =
          kernel == nullptr
              ? nullptr
              : static_cast<VulkanMapEncodeResources *>(kernel->resource.get());
      if (map == nullptr || map->prepared == nullptr ||
          !frame.proofs[stage].valid) {
        cleanup();
        return fail("accel_kernel_pipeline_invalid");
      }
      saved_maps[local][stage] = map;
      saved_states[local][stage] = save_map_state(*map);
    }
    saved[local] = true;
    auto *const first = saved_maps[local][0u];
    auto *const second = saved_maps[local][1u];
    if (first == nullptr || second == nullptr) {
      cleanup();
      return fail("accel_kernel_pipeline_invalid");
    }
    {
      MapModeGuard generated_first{*first};
      MapModeGuard generated_second{*second};
      const std::array<VulkanMapEncodeResources *, 2u> maps{first, second};
      bool bound = true;
      for (std::size_t stage = 0u; stage < maps.size(); ++stage) {
        VulkanMapEncodeResources &map = *maps[stage];
        map.generated_owner = static_cast<std::uint64_t>(
            reinterpret_cast<std::uintptr_t>(&pipeline));
        map.generated_plan = frame.proofs[stage].op_hash_hi;
        map.generated_token = frame.proofs[stage].graph_id_hi;
        map.generated_run = frame.proofs[stage].graph_id_lo;
        map.generated_slot = 0u;
        map.generated_stride = 1u;
        map.generated_first_control_generation = local;
        map.generated_control_generation_stride = 0u;
        map.generated_first_descriptor_generation = frame.generations[stage];
        map.generated_descriptor_generation_stride = 0u;
        bound =
            BindVulkanGeneratedMapAdmission(map, frame.gates[stage],
                                            pipeline.control.summary) &&
            map_owner_matches(map, frame.control_pipelines[stage],
                              frame.control_descriptors[stage],
                              frame.gates[stage], pipeline.control.summary) &&
            bound;
      }
      if (!bound) {
        cleanup();
        return fail("accel_kernel_pipeline_invalid");
      }
      VulkanCommand &command = temporary[local];
      const rund::AccelCheck created = CreateCommand(
          pipeline.adapter->device, pipeline.adapter->compute_queue_family,
          command, CommandKind::ReusablePrimary);
      if (!created.ok) {
        cleanup();
        return fail(created.reason);
      }
      made[local] = true;
      const VulkanPipelineRecordSlice slice{
          .first = local, .count = 1u, .open = false, .close = false};
      const rund::AccelCheck recorded =
          RecordVulkanPipeline(pipeline, command, CommandKind::ReusablePrimary,
                               false, nullptr, nullptr, &slice);
      if (!recorded.ok) {
        cleanup();
        return fail(recorded.reason);
      }
    }
    restore_maps();
  }
  for (std::size_t local = 0u; local < 2u; ++local) {
    selection.graph_sequence_frames[local].command = temporary[local];
    selection.graph_sequence_frames[local].command_ready = true;
    temporary[local] = {};
    made[local] = false;
  }
  selection.graph_sequence_command_count = 2u;
  selection.graph_generated_diagnostics.recorded_count = 4u;
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
