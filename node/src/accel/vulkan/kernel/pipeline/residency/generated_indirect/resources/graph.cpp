#include "../../../../../buffer/access.hpp"
#include "../../../../../buffer/create.hpp"

#include "../internal.hpp"
#include "../map.hpp"
#include "../map_state.hpp"

#include "../../../../../map/api.hpp"
#include "../../../../../map/admission.hpp"
#include "../../../../../map/encode/control.hpp"
#include "../../../../../map/source/control.hpp"
#include "../../../../../map/local.hpp"
#include "../../../prepare/record.hpp"
#include "../../sliding/internal.hpp"

#include <kernel/core/checked.hpp>

#include <array>
#include <cstdint>
#include <limits>

#include <rund/counter.hpp>

namespace rund::node::accel::detail::vulkan_generated_indirect_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace {

[[nodiscard]] const char *graph_generated_native_reason(
    const VulkanResidencyGraphGeneratedPredicate predicate) noexcept {
  switch (predicate) {
  case VulkanResidencyGraphGeneratedPredicate::Pipeline:
    return "accel_vulkan_graph_generated_pipeline";
  case VulkanResidencyGraphGeneratedPredicate::EntryIdentity:
    return "accel_vulkan_graph_generated_entry_identity";
  case VulkanResidencyGraphGeneratedPredicate::EntryShape:
    return "accel_vulkan_graph_generated_entry_shape";
  case VulkanResidencyGraphGeneratedPredicate::BindingIndex:
    return "accel_vulkan_graph_generated_binding_index";
  case VulkanResidencyGraphGeneratedPredicate::BindingRole:
    return "accel_vulkan_graph_generated_binding_role";
  case VulkanResidencyGraphGeneratedPredicate::BindingAlias:
    return "accel_vulkan_graph_generated_binding_alias";
  case VulkanResidencyGraphGeneratedPredicate::ControlBinding:
    return "accel_vulkan_graph_generated_control_binding";
  case VulkanResidencyGraphGeneratedPredicate::Semantic:
    return "accel_vulkan_graph_generated_semantic";
  case VulkanResidencyGraphGeneratedPredicate::None:
    break;
  }
  return "accel_vulkan_graph_generated_invalid";
}

[[nodiscard]] rund::AccelCheck
build_frame_proof(const VulkanPipeline &pipeline, const std::size_t local,
                  const VulkanResidencyGraphStageGeneratedProof &semantic,
                  VulkanResidencyGraphStageGeneratedProof &proof) noexcept {
  VulkanResidencyGraphStageGeneratedProof current{};
  if (!impl::eligible_entry(pipeline, local, current)) {
    proof = current;
    return {false,
            graph_generated_native_reason(current.first_failure_predicate)};
  }
  if (!impl::same_semantics(current, semantic)) {
    proof = current;
    proof.first_failure_predicate =
        VulkanResidencyGraphGeneratedPredicate::Semantic;
    proof.first_failure_ordinal = static_cast<std::uint32_t>(local);
    proof.first_failure_reason = "graph_generated_semantic";
    proof.valid = false;
    return {false, graph_generated_native_reason(
                       VulkanResidencyGraphGeneratedPredicate::Semantic)};
  }
  proof = current;
  return proof.valid
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, graph_generated_native_reason(
                                           proof.first_failure_predicate)};
}

[[nodiscard]] bool seed_graph_row(VulkanPipeline &pipeline,
                                  VulkanResidencyGraphGeneratedFrame &frame,
                                  const std::size_t local) noexcept {
  if (frame.gate.mapped == nullptr ||
      frame.gate.bytes < sizeof(VulkanResidencySlidingPayload) ||
      !frame.proof.valid || local >= PreparedPipelineStepCapacity) {
    return false;
  }
  VulkanResidencySlidingPayload payload{};
  const auto split = [](const std::uint64_t value) {
    return std::array<std::uint32_t, 2u>{
        static_cast<std::uint32_t>(value),
        static_cast<std::uint32_t>(value >> 32u)};
  };
  const auto owner = split(
      static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(&pipeline)));
  const auto plan = split(frame.proof.op_hash_hi);
  const auto token = split(frame.proof.graph_id_hi);
  const auto run = split(frame.proof.graph_id_lo);
  std::uint64_t turn = 0u;
  std::uint64_t coordinate = 0u;
  std::uint64_t control_generation = 0u;
  if (!rund::kernel::checked::mul(static_cast<std::uint64_t>(local), 1u,
                                  turn) ||
      !rund::kernel::checked::add(turn, 0u, coordinate) ||
      !rund::kernel::checked::add(0u, turn, control_generation)) {
    return false;
  }
  auto &row = payload.words;
  row[0u] = owner[0u];
  row[1u] = owner[1u];
  row[2u] = plan[0u];
  row[3u] = plan[1u];
  row[4u] = token[0u];
  row[5u] = token[1u];
  row[6u] = run[0u];
  row[7u] = run[1u];
  row[8u] = static_cast<std::uint32_t>(coordinate);
  row[9u] = static_cast<std::uint32_t>(coordinate >> 32u);
  row[10u] = static_cast<std::uint32_t>(turn);
  row[11u] = static_cast<std::uint32_t>(turn >> 32u);
  row[16u] = static_cast<std::uint32_t>(frame.expected_descriptor_generation);
  row[17u] =
      static_cast<std::uint32_t>(frame.expected_descriptor_generation >> 32u);
  row[20u] = 1u;
  row[21u] = 1u;
  row[22u] = 0u;
  row[24u] = frame.proof.local_count;
  row[25u] = static_cast<std::uint32_t>(control_generation);
  row[VulkanResidencySlidingLocalWord] = static_cast<std::uint32_t>(local);
  for (std::size_t index = 0u; index < VulkanResidencySlidingRowWords;
       ++index) {
    row[VulkanResidencySlidingRowWords + index] = row[index];
  }
  row[VulkanResidencySlidingAcceptedWord] = 0u;
  row[VulkanResidencySlidingReasonWord] = 0u;
  row[VulkanResidencySlidingObservedGenerationWord] =
      static_cast<std::uint32_t>(frame.expected_descriptor_generation);
  row[VulkanResidencySlidingObservedGenerationWord + 1u] =
      static_cast<std::uint32_t>(frame.expected_descriptor_generation >> 32u);
  return UploadVulkanBuffer(frame.gate, &payload, sizeof(payload));
}

} // namespace

namespace impl {

rund::AccelCheck prepare_graph(VulkanPipeline &pipeline,
                               VulkanResidencySelection &selection) noexcept {
  VulkanResidencySlidingGate &gate = selection.sliding;
  const std::size_t entry_count =
      pipeline.record == nullptr ? 0u : pipeline.record->entries.size();
  const auto fail = [&](const char *const reason) noexcept {
    auto &diagnostics = selection.graph_generated_diagnostics;
    diagnostics.phase = VulkanResidencyGraphGeneratedPhase::Quarantined;
    diagnostics.first_reason = reason;
    diagnostics.failed = true;
    diagnostics.quarantined = true;
    destroy_graph(selection);
    selection.graph_generated_phase =
        VulkanResidencyGraphGeneratedPhase::Quarantined;
    selection.quarantined.store(true, std::memory_order_release);
    diagnostics.phase = VulkanResidencyGraphGeneratedPhase::Quarantined;
    diagnostics.first_reason = reason;
    diagnostics.failed = true;
    diagnostics.quarantined = true;
    return rund::AccelCheck{false, reason};
  };
  if (selection.graph_generated_phase !=
          VulkanResidencyGraphGeneratedPhase::NotApplicable ||
      selection.graph_generated_generation != 0u) {
    selection.graph_generated_diagnostics.replay = true;
    return fail("accel_kernel_pipeline_replay");
  }
  if (pipeline.record == nullptr || selection.adapter != pipeline.adapter ||
      !selection.graph_generated.valid ||
      selection.graph_generated.local_count != entry_count ||
      entry_count == 0u || entry_count > PreparedPipelineStepCapacity) {
    return fail("accel_vulkan_pipeline_selection_unavailable");
  }
  if (!vulkan_sliding_detail::coherent_storage(pipeline.control.summary)) {
    return fail("accel_vulkan_memory_unavailable");
  }
  const rund::AccelCheck timeline =
      VulkanTimelineCapability(pipeline.adapter->timeline);
  if (!timeline.ok) {
    return fail(timeline.reason);
  }
  if (pipeline.adapter->timeline.signal == nullptr) {
    return fail("accel_vulkan_timeline_failed");
  }
  if (!rund::kernel::checked::add(0u, 1u,
                                  selection.graph_generated_generation) ||
      selection.graph_generated_generation == 0u) {
    selection.graph_generated_diagnostics.generation_overflow = true;
    return fail("compute_pipeline_capacity");
  }
  selection.graph_generated_phase =
      VulkanResidencyGraphGeneratedPhase::Admitted;
  auto &diagnostics = selection.graph_generated_diagnostics;
  diagnostics = {};
  diagnostics.phase = VulkanResidencyGraphGeneratedPhase::Admitted;
  diagnostics.first_reason = "ok";
  diagnostics.generation = selection.graph_generated_generation;
  diagnostics.admission_count = static_cast<std::uint32_t>(entry_count);
  try {
    gate.descriptor_leases.reserve(entry_count);
  } catch (...) {
    return fail("compute_pipeline_capacity");
  }
  VulkanLeaseScope lease_scope{*pipeline.adapter, gate.descriptor_leases};
  for (std::size_t local = 0u; local < entry_count; ++local) {
    auto &frame = selection.graph_generated_frames[local];
    frame.proof = {};
    const rund::AccelCheck proof_check = build_frame_proof(
        pipeline, local, selection.graph_generated, frame.proof);
    if (!proof_check.ok || !frame.proof.valid) {
      diagnostics.first_failure_predicate = frame.proof.first_failure_predicate;
      diagnostics.first_failure_ordinal = frame.proof.first_failure_ordinal;
      const char *const reason = proof_check.ok
                                     ? graph_generated_native_reason(
                                           frame.proof.first_failure_predicate)
                                     : proof_check.reason;
      return fail(reason);
    }
    const VulkanPipelineRecordEntry &record_entry =
        pipeline.record->entries[local];
    auto *const resources =
        static_cast<VulkanKernelResources *>(record_entry.prepared.get());
    auto *const kernel = resources == nullptr ? nullptr : resources->entry(0u);
    auto *const map =
        kernel == nullptr
            ? nullptr
            : static_cast<VulkanMapEncodeResources *>(kernel->resource.get());
    if (map == nullptr || map->prepared == nullptr) {
      return {false, "accel_kernel_pipeline_invalid"};
    }
    if (!rund::kernel::checked::add(static_cast<std::uint64_t>(local), 1u,
                                    frame.expected_descriptor_generation) ||
        (frame.generation = frame.expected_descriptor_generation) == 0u ||
        !CreateVulkanBuffer(*pipeline.adapter,
                            sizeof(VulkanResidencySlidingPayload),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, frame.gate) ||
        !vulkan_sliding_detail::coherent_storage(frame.gate) ||
        !ClearVulkanBuffer(frame.gate, frame.gate.bytes)) {
      return fail("accel_vulkan_memory_unavailable");
    }
    frame.proof.gate_generation = frame.expected_descriptor_generation;
    const auto artifact =
        VulkanGeneratedMapControlArtifact(map->prepared->plan);
    if (!artifact.ok || artifact.source_text.empty()) {
      return fail(artifact.reason);
    }
    map->generated_control_pipeline = AcquireVulkanCollectivePipeline(
        *pipeline.adapter, 6u, sizeof(VulkanGeneratedMapControlPush),
        map->prepared->plan, artifact);
    if (map->generated_control_pipeline == nullptr ||
        !ReserveVulkanCollectiveDescriptorDemand(
            *pipeline.adapter, *map->generated_control_pipeline, 6u, 1u) ||
        !AcquireVulkanCollectiveDescriptorSet(
            *pipeline.adapter, *map->generated_control_pipeline, 6u,
            map->generated_control_descriptor)) {
      return fail("accel_vulkan_descriptor_unavailable");
    }
    bool bound = false;
    {
      MapModeGuard generated{*map};
      bound = BindVulkanGeneratedMapAdmission(
          *map, frame.gate, pipeline.control.summary);
    }
    if (!bound || map->generated_gate_binding.buffer == VK_NULL_HANDLE ||
        map->generated_summary_binding.buffer == VK_NULL_HANDLE ||
        map->generated_gate_binding.bytes == 0u ||
        map->generated_summary_binding.bytes == 0u ||
        !seed_graph_row(pipeline, frame, local)) {
      return fail("accel_vulkan_descriptor_unavailable");
    }
    frame.control_pipeline = map->generated_control_pipeline;
    frame.control_descriptor = map->generated_control_descriptor;
    frame.ready = true;
    diagnostics.ready_count++;
  }
  if (diagnostics.ready_count != entry_count) {
    return fail("accel_vulkan_descriptor_unavailable");
  }
  gate.ready_for_submit = true;
  gate.retained_bytes = 0u;
  for (std::size_t local = 0u; local < entry_count; ++local) {
    gate.retained_bytes = ::rund::detail::counter::SaturatingAdd(
        gate.retained_bytes,
        selection.graph_generated_frames[local].gate.allocated_bytes);
  }
  selection.graph_generated_command_count = 0u;
  diagnostics.phase = VulkanResidencyGraphGeneratedPhase::Admitted;
  return {true, "ok"};
}

void destroy_graph(VulkanResidencySelection &selection) noexcept {
  if (selection.adapter == nullptr) {
    return;
  }
  for (auto &frame : selection.graph_generated_frames) {
    DestroyCommand(selection.adapter->device, frame.command);
    DestroyVulkanBuffer(*selection.adapter, frame.gate);
    frame = {};
  }
  selection.graph_generated_submitted.fill(false);
  selection.graph_generated_submitted_count = 0u;
  selection.graph_generated_command_count = 0u;
  ReleaseVulkanLeases(selection.sliding.descriptor_leases);
  DestroyVulkanBuffer(*selection.adapter, selection.sliding.descriptor);
  selection.sliding.ready_for_submit = false;
  if (selection.graph_generated_phase !=
          VulkanResidencyGraphGeneratedPhase::NotApplicable &&
      selection.graph_generated_phase !=
          VulkanResidencyGraphGeneratedPhase::TerminalKnown &&
      selection.graph_generated_phase !=
          VulkanResidencyGraphGeneratedPhase::Recorded) {
    selection.graph_generated_phase =
        VulkanResidencyGraphGeneratedPhase::Quarantined;
    selection.graph_generated_diagnostics.phase =
        VulkanResidencyGraphGeneratedPhase::Quarantined;
    selection.graph_generated_diagnostics.quarantined = true;
  }
}

} // namespace impl

#endif

} // namespace rund::node::accel::detail::vulkan_generated_indirect_detail
