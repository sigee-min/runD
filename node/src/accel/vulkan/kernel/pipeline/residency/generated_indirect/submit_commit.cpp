#include "internal.hpp"
#include "lifecycle.hpp"

#include <kernel/core/checked.hpp>

namespace rund::node::accel::detail::vulkan_generated_indirect_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace impl {

void commit_plan(VulkanPipeline &pipeline,
                 VulkanResidencySelection &selection,
                 const VulkanResidencyGraphSubmitPlan &plan) noexcept {
  const bool sequence =
      selection.mode == VulkanResidencyMode::GraphStageSequence;
  const std::size_t active_count = plan.active_count;
  std::size_t row_count = 0u;
  const bool row_count_valid =
      active_rows(selection.mode, active_count, row_count);
  const auto reject = [&]() noexcept {
    quarantine(pipeline, selection, "accel_kernel_pipeline_invalid",
               std::span<const bool>{plan.local_mask});
  };
  if (sequence) {
    std::uint64_t expected_seq{};
    std::uint64_t expected_total{};
    bool valid =
        row_count_valid && plan.valid && active_count >= 1u &&
        active_count <= 2u && plan.command_count == active_count + 2u &&
        selection.graph_generated_phase ==
            VulkanResidencyGraphGeneratedPhase::Recorded &&
        selection.graph_sequence_submitted_count == 0u &&
        rund::kernel::checked::add(selection.submit_seq, 1u, expected_seq) &&
        expected_seq == plan.next_submit_seq &&
        rund::kernel::checked::add(selection.submit_total, row_count,
                                   expected_total) &&
        expected_total == plan.next_submit_total && row_count != 0u;
    for (std::size_t local = 0u;
         valid && local < selection.graph_sequence.local_count; ++local) {
      if (!plan.local_mask[local]) {
        continue;
      }
      valid = selection.graph_sequence_frames[local].command_ready &&
              !selection.graph_sequence_frames[local].submitted &&
              !selection.graph_sequence_frames[local].quarantined;
    }
    valid = valid && plan.command_count == active_count + 2u &&
            plan.commands[0u] == selection.prefix.buffer &&
            plan.commands[active_count + 1u] == selection.suffix.buffer;
    for (std::size_t index = 0u; valid && index < plan.command_count; ++index) {
      valid = plan.commands[index] != VK_NULL_HANDLE;
      for (std::size_t prior = 0u; valid && prior < index; ++prior) {
        valid = plan.commands[prior] != plan.commands[index];
      }
    }
    if (!valid) {
      reject();
      return;
    }
    selection.submit_seq = plan.next_submit_seq;
    selection.submit_total = plan.next_submit_total;
    selection.graph_sequence_submitted.fill(false);
    for (std::size_t local = 0u;
         local < selection.graph_sequence_submitted.size(); ++local) {
      if (plan.local_mask[local]) {
        selection.graph_sequence_submitted[local] = true;
        selection.graph_sequence_frames[local].submitted = true;
      }
    }
    selection.graph_sequence_submitted_count = active_count;
    selection.graph_generated_diagnostics.submitted_count =
        static_cast<std::uint32_t>(row_count);
    selection.graph_generated_diagnostics.submission_generation =
        plan.next_submit_seq;
    selection.graph_generated_diagnostics.submit_seq = selection.submit_seq;
    selection.graph_generated_diagnostics.submit_total = selection.submit_total;
    selection.graph_generated_phase =
        VulkanResidencyGraphGeneratedPhase::Submitted;
    selection.graph_generated_diagnostics.phase =
        VulkanResidencyGraphGeneratedPhase::Submitted;
    return;
  }
  std::uint64_t expected_seq{};
  std::uint64_t expected_total{};
  if (!plan.valid ||
      selection.graph_generated_phase !=
          VulkanResidencyGraphGeneratedPhase::Recorded ||
      !row_count_valid || active_count == 0u ||
      active_count > selection.graph_generated.local_count ||
      plan.command_count != active_count + 2u ||
      !rund::kernel::checked::add(selection.submit_seq, 1u, expected_seq) ||
      expected_seq != plan.next_submit_seq ||
      !rund::kernel::checked::add(selection.submit_total,
                                  static_cast<std::uint64_t>(row_count),
                                  expected_total) ||
      expected_total != plan.next_submit_total) {
    reject();
    return;
  }
  if (active_count > selection.graph_generated_frames.size() ||
      plan.command_count != active_count + 2u ||
      selection.prefix.buffer == VK_NULL_HANDLE ||
      selection.suffix.buffer == VK_NULL_HANDLE ||
      selection.prefix.buffer == selection.suffix.buffer ||
      plan.commands[0u] != selection.prefix.buffer ||
      plan.commands[active_count + 1u] != selection.suffix.buffer) {
    reject();
    return;
  }
  for (std::size_t local = 0u; local < selection.graph_generated.local_count;
       ++local) {
    if (!plan.local_mask[local]) {
      continue;
    }
    if (!selection.graph_generated_frames[local].ready ||
        selection.graph_generated_frames[local].quarantined ||
        selection.graph_generated_frames[local].submitted) {
      reject();
      return;
    }
  }
  for (std::size_t index = 0u; index < plan.command_count; ++index) {
    if (plan.commands[index] == VK_NULL_HANDLE) {
      reject();
      return;
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (plan.commands[prior] == plan.commands[index]) {
        reject();
        return;
      }
    }
  }
  selection.submit_seq = plan.next_submit_seq;
  selection.submit_total = plan.next_submit_total;
  selection.graph_generated_submitted.fill(false);
  for (std::size_t local = 0u;
       local < selection.graph_generated_submitted.size(); ++local) {
    if (plan.local_mask[local]) {
      selection.graph_generated_submitted[local] = true;
      selection.graph_generated_frames[local].submitted = true;
    }
  }
  selection.graph_generated_submitted_count = active_count;
  selection.graph_generated_diagnostics.submitted_count =
      static_cast<std::uint32_t>(row_count);
  selection.graph_generated_diagnostics.submission_generation =
      plan.next_submit_seq;
  selection.graph_generated_diagnostics.submit_seq = selection.submit_seq;
  selection.graph_generated_diagnostics.submit_total = selection.submit_total;
  selection.graph_generated_phase =
      VulkanResidencyGraphGeneratedPhase::Submitted;
  selection.graph_generated_diagnostics.phase =
      VulkanResidencyGraphGeneratedPhase::Submitted;
}

} // namespace impl

void commit(VulkanPipeline &pipeline, const Plan &plan) noexcept {
  if (!plan.generated || pipeline.residency == nullptr || !plan.graph.valid) {
    return;
  }
  impl::commit_plan(pipeline, *pipeline.residency, plan.graph);
}

#endif

} // namespace rund::node::accel::detail::vulkan_generated_indirect_detail
