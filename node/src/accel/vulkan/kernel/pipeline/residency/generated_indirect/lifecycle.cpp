#include "lifecycle.hpp"
#include "internal.hpp"

#include "../../../../command/resources.hpp"
#include "../../prepare/record.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <limits>

namespace rund::node::accel::detail::vulkan_generated_indirect_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

bool add_totals(
    VulkanResidencySelection &selection,
    const VulkanResidencyGraphGeneratedDiagnostics &diagnostics) noexcept {
  std::uint64_t done_total{};
  std::uint64_t accept_total{};
  std::uint64_t known_total{};
  std::uint64_t unknown_total{};
  if (!rund::kernel::checked::add(selection.done_total,
                                  diagnostics.terminal_count, done_total) ||
      !rund::kernel::checked::add(selection.accept_total,
                                  diagnostics.accepted_count, accept_total) ||
      !rund::kernel::checked::add(selection.known_total,
                                  diagnostics.known_failure_count,
                                  known_total) ||
      !rund::kernel::checked::add(selection.unknown_total,
                                  diagnostics.unknown_count, unknown_total)) {
    return false;
  }
  selection.done_total = done_total;
  selection.accept_total = accept_total;
  selection.known_total = known_total;
  selection.unknown_total = unknown_total;
  return true;
}

void quarantine(VulkanPipeline &pipeline, VulkanResidencySelection &selection,
                const char *const reason,
                const std::span<const bool> pending) noexcept {
  const bool sequence =
      selection.mode == VulkanResidencyMode::GraphStageSequence;
  const bool replace_frontier = !pending.empty();
  std::size_t active_count = 0u;
  if (sequence) {
    if (replace_frontier) {
      selection.graph_sequence_submitted.fill(false);
      const std::size_t count = std::min(
          pending.size(), selection.graph_sequence_submitted.size());
      for (std::size_t local = 0u; local < count; ++local) {
        selection.graph_sequence_submitted[local] = pending[local];
      }
    }
    active_count = static_cast<std::size_t>(std::count(
        selection.graph_sequence_submitted.begin(),
        selection.graph_sequence_submitted.end(), true));
    selection.graph_sequence_submitted_count = active_count;
    for (std::size_t local = 0u;
         local < selection.graph_sequence_submitted.size(); ++local) {
      if (selection.graph_sequence_submitted[local]) {
        selection.graph_sequence_frames[local].submitted = true;
        selection.graph_sequence_frames[local].quarantined = true;
      }
    }
  } else if (selection.mode ==
             VulkanResidencyMode::GraphStageGeneratedIndirect) {
    if (replace_frontier) {
      selection.graph_generated_submitted.fill(false);
      const std::size_t count = std::min(
          pending.size(), selection.graph_generated_submitted.size());
      for (std::size_t local = 0u; local < count; ++local) {
        selection.graph_generated_submitted[local] = pending[local];
      }
    }
    active_count = static_cast<std::size_t>(std::count(
        selection.graph_generated_submitted.begin(),
        selection.graph_generated_submitted.end(), true));
    selection.graph_generated_submitted_count = active_count;
    for (std::size_t local = 0u;
         local < selection.graph_generated_submitted.size(); ++local) {
      if (selection.graph_generated_submitted[local]) {
        selection.graph_generated_frames[local].submitted = true;
        selection.graph_generated_frames[local].quarantined = true;
      }
    }
  }
  std::size_t row_count = 0u;
  (void)impl::active_rows(selection.mode, active_count, row_count);
  selection.graph_generated_phase =
      VulkanResidencyGraphGeneratedPhase::TerminalUnknown;
  selection.graph_generated_diagnostics.phase =
      VulkanResidencyGraphGeneratedPhase::TerminalUnknown;
  selection.graph_generated_diagnostics.first_reason =
      reason == nullptr ? "accel_kernel_pipeline_invalid" : reason;
  selection.graph_generated_diagnostics.failed = true;
  selection.graph_generated_diagnostics.quarantined = true;
  selection.graph_generated_diagnostics.submitted_count =
      static_cast<std::uint32_t>(row_count);
  selection.graph_generated_diagnostics.unknown_count =
      static_cast<std::uint32_t>(row_count);
  selection.quarantined.store(true, std::memory_order_release);
  if (pipeline.adapter != nullptr) {
    pipeline.adapter->residency_quarantined.store(true,
                                                  std::memory_order_release);
  }
}

namespace {

void reset_cycle_diagnostics(VulkanResidencySelection &selection) noexcept {
  selection.graph_generated_phase =
      VulkanResidencyGraphGeneratedPhase::Recorded;
  selection.graph_generated_diagnostics.phase =
      VulkanResidencyGraphGeneratedPhase::Recorded;
  selection.graph_generated_diagnostics.first_reason = "ok";
  selection.graph_generated_diagnostics.submission_generation = 0u;
  selection.graph_generated_diagnostics.submitted_count = 0u;
  selection.graph_generated_diagnostics.terminal_count = 0u;
  selection.graph_generated_diagnostics.accepted_count = 0u;
  selection.graph_generated_diagnostics.known_failure_count = 0u;
  selection.graph_generated_diagnostics.unknown_count = 0u;
  selection.graph_generated_diagnostics.failed = false;
  selection.graph_generated_diagnostics.quarantined = false;
  selection.graph_generated_diagnostics.submit_seq = selection.submit_seq;
  selection.graph_generated_diagnostics.done_seq = selection.done_seq;
  selection.graph_generated_diagnostics.submit_total = selection.submit_total;
  selection.graph_generated_diagnostics.done_total = selection.done_total;
  selection.graph_generated_diagnostics.accept_total = selection.accept_total;
  selection.graph_generated_diagnostics.known_total = selection.known_total;
  selection.graph_generated_diagnostics.unknown_total =
      selection.unknown_total;
}

[[nodiscard]] bool advance_expected_generation(
    VulkanPipeline &pipeline) noexcept {
  std::uint64_t next_expected{};
  if (!rund::kernel::checked::add(pipeline.expected_control_generation, 1u,
                                  next_expected) ||
      next_expected > rund::compute::PipelineGenerationCapacity ||
      next_expected > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  pipeline.expected_control_generation = next_expected;
  return true;
}

} // namespace

void close_known(VulkanPipeline &pipeline, VulkanResidencySelection &selection,
                 const char *const reason) noexcept {
  selection.graph_generated_phase =
      VulkanResidencyGraphGeneratedPhase::TerminalKnown;
  selection.graph_generated_diagnostics.phase =
      VulkanResidencyGraphGeneratedPhase::TerminalKnown;
  selection.graph_generated_diagnostics.first_reason = reason;
  selection.graph_generated_diagnostics.failed = true;
  if (selection.mode == VulkanResidencyMode::GraphStageGeneratedIndirect) {
    if (pipeline.adapter != nullptr) {
      for (auto &frame : selection.graph_generated_frames) {
        DestroyCommand(pipeline.adapter->device, frame.command);
        frame.command = {};
        frame.ready = false;
        frame.submitted = false;
        frame.quarantined = false;
      }
    }
    selection.graph_generated_submitted.fill(false);
    selection.graph_generated_submitted_count = 0u;
    selection.graph_generated_command_count = 0u;
    selection.sliding.ready_for_submit = false;
    return;
  }
  if (selection.mode == VulkanResidencyMode::GraphStageSequence) {
    impl::destroy_sequence(selection);
  }
}

bool rearm(VulkanPipeline &pipeline,
           VulkanResidencySelection &selection) noexcept {
  if (!advance_expected_generation(pipeline)) {
    return false;
  }
  reset_cycle_diagnostics(selection);
  if (selection.mode == VulkanResidencyMode::GraphStageGeneratedIndirect) {
    selection.graph_generated_submitted.fill(false);
    selection.graph_generated_submitted_count = 0u;
    selection.graph_generated_command_count =
        selection.graph_generated.local_count;
    for (auto &frame : selection.graph_generated_frames) {
      frame.submitted = false;
      frame.quarantined = false;
    }
  } else if (selection.mode == VulkanResidencyMode::GraphStageSequence) {
    selection.graph_sequence_submitted.fill(false);
    selection.graph_sequence_submitted_count = 0u;
    selection.graph_sequence_command_count = 2u;
    for (auto &frame : selection.graph_sequence_frames) {
      frame.submitted = false;
      frame.quarantined = false;
    }
  } else {
    return false;
  }
  selection.quarantined.store(false, std::memory_order_release);
  return true;
}

#endif

} // namespace rund::node::accel::detail::vulkan_generated_indirect_detail
