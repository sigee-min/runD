#include "../internal.hpp"
#include "../lifecycle.hpp"
#include "../map.hpp"

#include "../../../../ops/status.hpp"
#include "../../sliding/internal.hpp"

#include "../../../../../../kernel/prepared/interface/api.hpp"
#include "../../../../../../kernel/prepared/model.hpp"
#include "../../../prepare/record.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <limits>

namespace rund::node::accel::detail::vulkan_generated_indirect_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace {

[[nodiscard]] bool
can_rearm(const VulkanResidencySelection &selection,
          const VulkanResidencyGraphGeneratedDiagnostics &diagnostics,
          const bool proof_match, const bool sequence_match) noexcept {
  const std::uint32_t active_count = diagnostics.submitted_count;
  return proof_match && sequence_match &&
         selection.done_seq == selection.submit_seq && active_count != 0u &&
         diagnostics.accepted_count == active_count &&
         diagnostics.terminal_count == active_count &&
         diagnostics.submitted_count == active_count &&
         diagnostics.accepted_count == diagnostics.terminal_count &&
         diagnostics.terminal_count == diagnostics.submitted_count &&
         diagnostics.known_failure_count == 0u &&
         diagnostics.unknown_count == 0u;
}

} // namespace

bool finish(VulkanPipeline &pipeline, KernelResult &result) noexcept {
  if (pipeline.residency == nullptr ||
      (pipeline.residency->mode !=
           VulkanResidencyMode::GraphStageGeneratedIndirect &&
       pipeline.residency->mode != VulkanResidencyMode::GraphStageSequence)) {
    return false;
  }
  if (pipeline.residency->mode ==
      VulkanResidencyMode::GraphStageGeneratedIndirect) {
    VulkanResidencySelection &selection = *pipeline.residency;
    auto &diagnostics = selection.graph_generated_diagnostics;
    diagnostics.accepted_count = 0u;
    diagnostics.known_failure_count = 0u;
    diagnostics.unknown_count = 0u;
    diagnostics.terminal_count = 0u;
    const std::uint32_t capacity = selection.graph_generated.local_count;
    const std::uint32_t active_count =
        static_cast<std::uint32_t>(selection.graph_generated_submitted_count);
    std::size_t row_count = 0u;
    const bool row_shape =
        impl::active_rows(VulkanResidencyMode::GraphStageGeneratedIndirect,
                          active_count, row_count);
    bool post_submit_unknown = !result.check.ok;
    bool known_failure = false;
    const char *known_reason = result.check.reason;
    if (!result.check.ok) {
      diagnostics.first_reason = result.check.reason;
    }
    std::uint64_t next_done_seq{};
    const bool sequence_match =
        rund::kernel::checked::add(selection.done_seq, 1u, next_done_seq) &&
        next_done_seq == selection.submit_seq;
    const bool done_sequence_advanced =
        rund::kernel::checked::add(selection.done_seq, 1u, next_done_seq);
    if (done_sequence_advanced) {
      selection.done_seq = next_done_seq;
    } else {
      post_submit_unknown = true;
    }
    const bool proof_match =
        result.check.ok && impl::graph_proof_matches(pipeline, selection);
    if (!proof_match) {
      post_submit_unknown = true;
      diagnostics.first_reason = "accel_kernel_pipeline_invalid";
    }
    const bool control_valid =
        result.pipeline.control_observed && pipeline.record != nullptr &&
        ValidPreparedPipelineControl(result.pipeline.control,
                                     pipeline.record->status) &&
        PreparedPipelineGenerationMatches(result.pipeline.control,
                                          pipeline.expected_control_generation);
    if (!control_valid) {
      post_submit_unknown = true;
      diagnostics.first_reason = "accel_kernel_pipeline_invalid";
    }
    const bool submitted_shape =
        row_shape && diagnostics.submitted_count == row_count;
    if (!submitted_shape || !sequence_match) {
      post_submit_unknown = true;
      diagnostics.first_reason = "accel_kernel_pipeline_replay";
    }
    for (std::size_t local = 0u; local < capacity; ++local) {
      auto &frame = selection.graph_generated_frames[local];
      if (!selection.graph_generated_submitted[local]) {
        continue;
      }
      diagnostics.terminal_count++;
      if (!result.check.ok) {
        continue;
      }
      VulkanResidencySlidingGate observed{};
      observed.descriptor = frame.gate;
      bool frame_known_failure = false;
      const char *frame_reason = "accel_kernel_pipeline_invalid";
      const bool frame_identity =
          frame.ready && frame.generation != 0u &&
          frame.generation == frame.expected_descriptor_generation &&
          frame.proof.valid && !frame.quarantined;
      const bool accepted =
          frame_identity &&
          gate_result(observed, frame.expected_descriptor_generation,
                      frame_known_failure, frame_reason);
      if (accepted) {
        diagnostics.accepted_count++;
      } else if (frame_known_failure) {
        diagnostics.known_failure_count++;
        if (!known_failure) {
          known_reason = frame_reason;
        }
        known_failure = true;
      } else {
        diagnostics.unknown_count++;
        post_submit_unknown = true;
        diagnostics.first_reason = "accel_kernel_pipeline_invalid";
        frame.quarantined = true;
      }
    }
    const std::uint64_t classified =
        static_cast<std::uint64_t>(diagnostics.accepted_count) +
        static_cast<std::uint64_t>(diagnostics.known_failure_count) +
        static_cast<std::uint64_t>(diagnostics.unknown_count);
    const bool complete_terminal =
        diagnostics.terminal_count == active_count &&
        diagnostics.submitted_count == active_count &&
        classified == diagnostics.submitted_count;
    if (diagnostics.unknown_count != 0u ||
        (diagnostics.accepted_count != 0u &&
         diagnostics.known_failure_count != 0u) ||
        (!complete_terminal && !known_failure)) {
      post_submit_unknown = true;
    }
    if (post_submit_unknown) {
      diagnostics.accepted_count = 0u;
      diagnostics.known_failure_count = 0u;
      diagnostics.unknown_count = active_count;
      for (std::size_t local = 0u; local < capacity; ++local) {
        if (selection.graph_generated_submitted[local]) {
          selection.graph_generated_frames[local].submitted = true;
          selection.graph_generated_frames[local].quarantined = true;
        }
      }
    }
    if (!add_totals(selection, diagnostics)) {
      post_submit_unknown = true;
      diagnostics.first_reason = "accel_kernel_pipeline_invalid";
      diagnostics.accepted_count = 0u;
      diagnostics.known_failure_count = 0u;
      diagnostics.unknown_count = active_count;
      for (std::size_t local = 0u; local < capacity; ++local) {
        if (selection.graph_generated_submitted[local]) {
          selection.graph_generated_frames[local].submitted = true;
          selection.graph_generated_frames[local].quarantined = true;
        }
      }
    }
    diagnostics.submit_seq = selection.submit_seq;
    diagnostics.done_seq = selection.done_seq;
    diagnostics.submit_total = selection.submit_total;
    diagnostics.done_total = selection.done_total;
    diagnostics.accept_total = selection.accept_total;
    diagnostics.known_total = selection.known_total;
    diagnostics.unknown_total = selection.unknown_total;
    const bool known_close = !post_submit_unknown && proof_match &&
                             sequence_match && known_failure &&
                             complete_terminal &&
                             diagnostics.accepted_count == 0u &&
                             diagnostics.known_failure_count == active_count &&
                             diagnostics.unknown_count == 0u;
    const bool should_rearm =
        !post_submit_unknown && !known_failure && result.check.ok &&
        can_rearm(selection, diagnostics, proof_match, sequence_match);
    if (should_rearm) {
      if (rearm(pipeline, selection)) {
        result.terminal = NativeTerminal::Known;
      } else {
        diagnostics.first_reason = "accel_kernel_pipeline_replay";
        result.check = rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
        result.terminal = NativeTerminal::UnknownMayWrite;
        quarantine(pipeline, selection, diagnostics.first_reason);
      }
    } else if (known_close) {
      result.check = rund::AccelCheck{false, known_reason};
      result.terminal = NativeTerminal::Known;
      close_known(pipeline, selection, known_reason);
    } else {
      result.check = rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
      result.terminal = NativeTerminal::UnknownMayWrite;
      quarantine(pipeline, selection, diagnostics.first_reason);
    }
    return true;
  }

  VulkanResidencySelection &selection = *pipeline.residency;
  auto &diagnostics = selection.graph_generated_diagnostics;
  diagnostics.accepted_count = 0u;
  diagnostics.known_failure_count = 0u;
  diagnostics.unknown_count = 0u;
  diagnostics.terminal_count = 0u;
  const std::size_t active_count = selection.graph_sequence_submitted_count;
  std::size_t row_count = 0u;
  const bool row_shape = impl::active_rows(
      VulkanResidencyMode::GraphStageSequence, active_count, row_count);
  bool post_submit_unknown = !result.check.ok;
  bool known_failure = false;
  const char *known_reason = result.check.reason;
  std::uint64_t next_done_seq{};
  const bool sequence_match =
      rund::kernel::checked::add(selection.done_seq, 1u, next_done_seq) &&
      next_done_seq == selection.submit_seq;
  if (rund::kernel::checked::add(selection.done_seq, 1u, next_done_seq)) {
    selection.done_seq = next_done_seq;
  } else {
    post_submit_unknown = true;
  }
  const bool proof_match =
      result.check.ok && impl::sequence_proof_matches(pipeline, selection);
  if (!proof_match) {
    post_submit_unknown = true;
    diagnostics.first_reason = "accel_kernel_pipeline_invalid";
  }
  const bool control_valid =
      result.pipeline.control_observed && pipeline.record != nullptr &&
      ValidPreparedPipelineControl(result.pipeline.control,
                                   pipeline.record->status) &&
      PreparedPipelineGenerationMatches(result.pipeline.control,
                                        pipeline.expected_control_generation);
  if (!control_valid) {
    post_submit_unknown = true;
    diagnostics.first_reason = "accel_kernel_pipeline_invalid";
  }
  const bool submitted_shape =
      row_shape && diagnostics.submitted_count == row_count;
  if (!submitted_shape || !sequence_match) {
    post_submit_unknown = true;
    diagnostics.first_reason = "accel_kernel_pipeline_replay";
  }
  for (std::size_t local = 0u; local < selection.graph_sequence.local_count;
       ++local) {
    auto &frame = selection.graph_sequence_frames[local];
    if (!selection.graph_sequence_submitted[local]) {
      continue;
    }
    for (std::size_t stage = 0u; stage < 2u; ++stage) {
      ++diagnostics.terminal_count;
      if (!result.check.ok) {
        continue;
      }
      VulkanResidencySlidingGate observed{};
      observed.descriptor = frame.gates[stage];
      bool frame_known_failure = false;
      const char *frame_reason = "accel_kernel_pipeline_invalid";
      const bool frame_identity =
          frame.ready[stage] && frame.generations[stage] != 0u &&
          frame.proofs[stage].valid && !frame.quarantined;
      const bool accepted =
          frame_identity && gate_result(observed, frame.generations[stage],
                                        frame_known_failure, frame_reason);
      if (accepted) {
        ++diagnostics.accepted_count;
      } else if (frame_known_failure) {
        ++diagnostics.known_failure_count;
        if (!known_failure) {
          known_reason = frame_reason;
        }
        known_failure = true;
      } else {
        ++diagnostics.unknown_count;
        post_submit_unknown = true;
        diagnostics.first_reason = "accel_kernel_pipeline_invalid";
        frame.quarantined = true;
      }
    }
  }
  const std::size_t classified = diagnostics.accepted_count +
                                 diagnostics.known_failure_count +
                                 diagnostics.unknown_count;
  const bool complete_terminal = diagnostics.terminal_count == row_count &&
                                 diagnostics.submitted_count == row_count &&
                                 classified == row_count;
  if (!complete_terminal ||
      (diagnostics.accepted_count != 0u &&
       diagnostics.known_failure_count != 0u) ||
      diagnostics.unknown_count != 0u) {
    post_submit_unknown = true;
  }
  if (post_submit_unknown) {
    diagnostics.accepted_count = 0u;
    diagnostics.known_failure_count = 0u;
    diagnostics.unknown_count = row_count;
    for (std::size_t local = 0u; local < selection.graph_sequence.local_count;
         ++local) {
      if (selection.graph_sequence_submitted[local]) {
        selection.graph_sequence_frames[local].submitted = true;
        selection.graph_sequence_frames[local].quarantined = true;
      }
    }
  }
  if (!add_totals(selection, diagnostics)) {
    post_submit_unknown = true;
    diagnostics.first_reason = "accel_kernel_pipeline_invalid";
    diagnostics.accepted_count = 0u;
    diagnostics.known_failure_count = 0u;
    diagnostics.unknown_count = row_count;
  }
  diagnostics.submit_seq = selection.submit_seq;
  diagnostics.done_seq = selection.done_seq;
  diagnostics.submit_total = selection.submit_total;
  diagnostics.done_total = selection.done_total;
  diagnostics.accept_total = selection.accept_total;
  diagnostics.known_total = selection.known_total;
  diagnostics.unknown_total = selection.unknown_total;
  const bool known_close =
      !post_submit_unknown && proof_match && sequence_match && known_failure &&
      complete_terminal && diagnostics.accepted_count == 0u &&
      diagnostics.known_failure_count == row_count &&
      diagnostics.unknown_count == 0u;
  const bool should_rearm =
      !post_submit_unknown && !known_failure && result.check.ok &&
      complete_terminal && diagnostics.accepted_count == row_count &&
      diagnostics.known_failure_count == 0u && diagnostics.unknown_count == 0u;
  if (should_rearm) {
    if (rearm(pipeline, selection)) {
      result.terminal = NativeTerminal::Known;
    } else {
      diagnostics.first_reason = "accel_kernel_pipeline_replay";
      result.check = rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
      result.terminal = NativeTerminal::UnknownMayWrite;
      quarantine(pipeline, selection, diagnostics.first_reason);
    }
  } else if (known_close) {
    result.check = rund::AccelCheck{false, known_reason};
    result.terminal = NativeTerminal::Known;
    close_known(pipeline, selection, known_reason);
  } else {
    result.check = rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    result.terminal = NativeTerminal::UnknownMayWrite;
    quarantine(pipeline, selection, diagnostics.first_reason);
  }
  return true;
}

#endif

} // namespace rund::node::accel::detail::vulkan_generated_indirect_detail
