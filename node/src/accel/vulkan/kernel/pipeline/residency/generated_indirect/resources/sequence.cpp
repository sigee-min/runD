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

[[nodiscard]] bool seed_sequence_gate(VulkanPipeline &pipeline,
                                      VulkanResidencyGraphSequenceFrame &frame,
                                      const std::size_t stage,
                                      const std::size_t local) noexcept {
  if (stage >= 2u || frame.gates[stage].mapped == nullptr ||
      frame.gates[stage].bytes < sizeof(VulkanResidencySlidingPayload) ||
      !frame.proofs[stage].valid) {
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
  const auto plan = split(frame.proofs[stage].op_hash_hi);
  const auto token = split(frame.proofs[stage].graph_id_hi);
  const auto run = split(frame.proofs[stage].graph_id_lo);
  const std::uint64_t coordinate = local;
  const std::uint64_t turn = local;
  const std::uint64_t generation = frame.generations[stage];
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
  row[16u] = static_cast<std::uint32_t>(generation);
  row[17u] = static_cast<std::uint32_t>(generation >> 32u);
  row[20u] = 1u;
  row[21u] = 1u;
  row[22u] = 0u;
  row[24u] = frame.proofs[stage].local_count;
  row[25u] = static_cast<std::uint32_t>(local);
  row[VulkanResidencySlidingLocalWord] = static_cast<std::uint32_t>(local);
  for (std::size_t index = 0u; index < VulkanResidencySlidingRowWords;
       ++index) {
    row[VulkanResidencySlidingRowWords + index] = row[index];
  }
  row[VulkanResidencySlidingAcceptedWord] = 0u;
  row[VulkanResidencySlidingReasonWord] = 0u;
  row[VulkanResidencySlidingObservedGenerationWord] =
      static_cast<std::uint32_t>(generation);
  row[VulkanResidencySlidingObservedGenerationWord + 1u] =
      static_cast<std::uint32_t>(generation >> 32u);
  return UploadVulkanBuffer(frame.gates[stage], &payload, sizeof(payload));
}

} // namespace

namespace impl {

rund::AccelCheck
prepare_sequence(VulkanPipeline &pipeline,
                 VulkanResidencySelection &selection) noexcept {
  auto &diagnostics = selection.graph_generated_diagnostics;
  const auto fail = [&](const char *const reason) noexcept {
    diagnostics.phase = VulkanResidencyGraphGeneratedPhase::Quarantined;
    diagnostics.first_reason = reason;
    diagnostics.failed = true;
    diagnostics.quarantined = true;
    selection.graph_generated_phase =
        VulkanResidencyGraphGeneratedPhase::Quarantined;
    selection.quarantined.store(true, std::memory_order_release);
    return rund::AccelCheck{false, reason};
  };
  if (selection.graph_generated_phase !=
          VulkanResidencyGraphGeneratedPhase::NotApplicable ||
      selection.graph_generated_generation != 0u) {
    diagnostics.replay = true;
    return fail("accel_kernel_pipeline_replay");
  }
  if (!selection.graph_sequence.valid ||
      selection.graph_sequence.stage_count != 2u ||
      selection.graph_sequence.local_count != 2u ||
      selection.adapter != pipeline.adapter || pipeline.record == nullptr) {
    return fail("accel_vulkan_pipeline_invalid");
  }
  VulkanResidencyGraphStageSequenceProof current{};
  if (!eligible_sequence(pipeline, current) ||
      current.stage_count != selection.graph_sequence.stage_count ||
      current.local_count != selection.graph_sequence.local_count) {
    return fail("accel_kernel_pipeline_invalid");
  }
  for (std::size_t stage = 0u; stage < 2u; ++stage) {
    auto expected = current.steps[stage];
    auto stored = selection.graph_sequence.steps[stage];
    expected.gate_generation = 0u;
    stored.gate_generation = 0u;
    if (!same_frame(expected, stored)) {
      return fail("accel_kernel_pipeline_invalid");
    }
  }
  for (std::size_t local = 0u; local < 2u; ++local) {
    for (std::size_t stage = 0u; stage < 2u; ++stage) {
      auto expected = current.frame_proofs[local][stage];
      auto stored = selection.graph_sequence.frame_proofs[local][stage];
      expected.gate_generation = 0u;
      stored.gate_generation = 0u;
      if (!same_frame(expected, stored)) {
        return fail("accel_kernel_pipeline_invalid");
      }
    }
  }
  const rund::AccelCheck timeline =
      VulkanTimelineCapability(pipeline.adapter->timeline);
  if (!timeline.ok) {
    return fail(timeline.reason);
  }
  if (!rund::kernel::checked::add(0u, 1u,
                                  selection.graph_generated_generation) ||
      selection.graph_generated_generation == 0u) {
    return fail("compute_pipeline_capacity");
  }
  diagnostics = {};
  diagnostics.phase = VulkanResidencyGraphGeneratedPhase::Admitted;
  diagnostics.first_reason = "ok";
  diagnostics.generation = selection.graph_generated_generation;
  diagnostics.admission_count = 4u;
  try {
    selection.sliding.descriptor_leases.reserve(4u);
  } catch (...) {
    return fail("compute_pipeline_capacity");
  }
  VulkanLeaseScope lease_scope{*pipeline.adapter,
                               selection.sliding.descriptor_leases};
  for (std::size_t local = 0u; local < 2u; ++local) {
    auto &frame = selection.graph_sequence_frames[local];
    frame = {};
    frame.proofs = current.frame_proofs[local];
    for (std::size_t stage = 0u; stage < 2u; ++stage) {
      std::uint64_t generation{};
      if (!rund::kernel::checked::add(
              static_cast<std::uint64_t>(local * 2u + stage), 1u, generation) ||
          !CreateVulkanBuffer(
              *pipeline.adapter, sizeof(VulkanResidencySlidingPayload),
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, frame.gates[stage]) ||
          !vulkan_sliding_detail::coherent_storage(frame.gates[stage]) ||
          !ClearVulkanBuffer(frame.gates[stage], frame.gates[stage].bytes)) {
        return fail("accel_vulkan_memory_unavailable");
      }
      frame.generations[stage] = generation;
      frame.ready[stage] = true;
      const auto &record_entry = pipeline.record->entries[local];
      auto *const resources =
          static_cast<VulkanKernelResources *>(record_entry.prepared.get());
      auto *const kernel =
          resources == nullptr ? nullptr : resources->entry(stage);
      auto *const map =
          kernel == nullptr
              ? nullptr
              : static_cast<VulkanMapEncodeResources *>(kernel->resource.get());
      if (map == nullptr || map->prepared == nullptr) {
        return fail("accel_kernel_pipeline_invalid");
      }
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
              frame.control_descriptors[stage])) {
        return fail("accel_vulkan_descriptor_unavailable");
      }
      map->generated_control_descriptor = frame.control_descriptors[stage];
      bool bound = false;
      {
        MapModeGuard generated{*map};
        bound = BindVulkanGeneratedMapAdmission(
            *map, frame.gates[stage], pipeline.control.summary);
      }
      if (!bound || !seed_sequence_gate(pipeline, frame, stage, local)) {
        return fail("accel_vulkan_descriptor_unavailable");
      }
      frame.control_pipelines[stage] = map->generated_control_pipeline;
      diagnostics.ready_count++;
    }
    std::size_t pin = 0u;
    for (std::size_t binding = 0u; binding < frame.proofs[0u].binding_count;
         ++binding) {
      if (frame.proofs[0u].binding_roles[binding] ==
              static_cast<std::uint8_t>(
                  rund::kernel::ComputeBindingAccess::Write) &&
          pin < frame.intermediate_pins.size()) {
        frame.intermediate_pins[pin++] =
            frame.proofs[0u].binding_handles[binding];
      }
    }
  }
  selection.graph_sequence_command_count = 0u;
  selection.graph_generated_phase =
      VulkanResidencyGraphGeneratedPhase::Admitted;
  return {true, "ok"};
}

void destroy_sequence(VulkanResidencySelection &selection) noexcept {
  if (selection.adapter == nullptr) {
    return;
  }
  for (auto &frame : selection.graph_sequence_frames) {
    DestroyCommand(selection.adapter->device, frame.command);
    for (auto &gate : frame.gates) {
      DestroyVulkanBuffer(*selection.adapter, gate);
    }
    frame = {};
  }
  selection.graph_sequence_submitted.fill(false);
  selection.graph_sequence_submitted_count = 0u;
  selection.graph_sequence_command_count = 0u;
  ReleaseVulkanLeases(selection.sliding.descriptor_leases);
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
