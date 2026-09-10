#include "match.hpp"

#include <kernel/program/compute/backend.hpp>

#include <cstddef>
#include <cstdint>

#include <kernel/core/checked.hpp>

namespace rund::node::accel::detail::vulkan_generated_indirect_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)


namespace impl {

[[nodiscard]] const BackendRun *
pipeline_run(const VulkanPipelineRecordEntry &entry) noexcept {
  return entry.run;
}

[[nodiscard]] const BoundStep *
entry_step(const VulkanPipelineRecordEntry &entry,
           const BackendRun *const run) noexcept {
  return run == nullptr || run->steps == nullptr || entry.step_index != 0u ||
                 run->step_count != 1u
             ? nullptr
             : &run->steps[entry.step_index];
}

[[nodiscard]] bool
copy_binding(const rund::kernel::ResidentBufferRef *const ref,
             const std::shared_ptr<void> *const handle, const std::size_t index,
             VulkanResidencyGraphStageGeneratedProof &proof) noexcept {
  if (index >= proof.binding_ids.size() || ref == nullptr ||
      handle == nullptr || *handle == nullptr || ref->id == 0u ||
      ref->bytes == 0u || ref->count == 0u ||
      ref->element_bytes != sizeof(std::uint64_t) ||
      ref->stride_bytes < ref->element_bytes) {
    return false;
  }
  proof.binding_ids[index] = ref->id;
  proof.binding_bytes[index] = ref->bytes;
  proof.binding_offsets[index] = ref->offset_bytes;
  proof.binding_elements[index] = ref->element_bytes;
  proof.binding_strides[index] = ref->stride_bytes;
  proof.binding_counts[index] = ref->count;
  proof.binding_handles[index] = *handle;
  return true;
}

[[nodiscard]] bool
same_resident_storage(const rund::kernel::ResidentBufferRef &left,
             const std::shared_ptr<void> *const left_handle,
             const rund::kernel::ResidentBufferRef &right,
             const std::shared_ptr<void> *const right_handle) noexcept {
  return left.id == right.id && left.bytes == right.bytes &&
         left.offset_bytes == right.offset_bytes &&
         left.element_bytes == right.element_bytes &&
         left.stride_bytes == right.stride_bytes && left.count == right.count &&
         left_handle != nullptr && right_handle != nullptr &&
         *left_handle != nullptr && *right_handle != nullptr &&
         left_handle->get() == right_handle->get();
}

[[nodiscard]] bool graph_alias_matches(
    const KernelExecution &execution, const RunBinds &source,
    const std::uint64_t graph_binding,
    const rund::kernel::ResidentBufferRef &resident,
    const std::shared_ptr<void> *const resident_handle) noexcept {
  if (graph_binding >= source.size() ||
      graph_binding >= execution.graph_roles.size() ||
      graph_binding >= execution.graph_alias_representatives.size()) {
    return false;
  }
  const std::uint64_t representative =
      execution.graph_alias_representatives[graph_binding];
  if (representative >= source.size() || representative > graph_binding ||
      !same_resident_storage(
          source.refs()[graph_binding], source.handles() + graph_binding,
          source.refs()[representative], source.handles() + representative)) {
    return false;
  }
  return same_resident_storage(resident, resident_handle, source.refs()[representative],
                      source.handles() + representative);
}



[[nodiscard]] bool
same_proof_storage(const VulkanResidencyGraphStageGeneratedProof &left,
             const std::size_t left_index,
             const VulkanResidencyGraphStageGeneratedProof &right,
             const std::size_t right_index) noexcept {
  return left.binding_ids[left_index] == right.binding_ids[right_index] &&
         left.binding_bytes[left_index] == right.binding_bytes[right_index] &&
         left.binding_offsets[left_index] ==
             right.binding_offsets[right_index] &&
         left.binding_elements[left_index] ==
             right.binding_elements[right_index] &&
         left.binding_strides[left_index] ==
             right.binding_strides[right_index] &&
         left.binding_counts[left_index] == right.binding_counts[right_index] &&
         left.binding_handles[left_index].get() ==
             right.binding_handles[right_index].get();
}



bool same_semantics(
    const VulkanResidencyGraphStageGeneratedProof &left,
    const VulkanResidencyGraphStageGeneratedProof &right) noexcept {
  if (left.kernel_id != right.kernel_id ||
      left.graph_id_hi != right.graph_id_hi ||
      left.graph_id_lo != right.graph_id_lo ||
      left.node_count != right.node_count ||
      left.op_hash_hi != right.op_hash_hi ||
      left.op_hash_lo != right.op_hash_lo ||
      left.binding_count != right.binding_count ||
      left.read_count != right.read_count ||
      left.write_count != right.write_count ||
      left.local_count != right.local_count ||
      left.count_binding != right.count_binding ||
      left.predicate_binding != right.predicate_binding ||
      left.count_offset != right.count_offset ||
      left.predicate_offset != right.predicate_offset ||
      left.count_capacity != right.count_capacity ||
      left.predicate_expected != right.predicate_expected ||
      left.count_source != right.count_source ||
      left.predicate_source != right.predicate_source ||
      left.has_count != right.has_count ||
      left.has_predicate != right.has_predicate) {
    return false;
  }
  for (std::size_t binding = 0u; binding < left.binding_count; ++binding) {
    if (left.binding_roles[binding] != right.binding_roles[binding] ||
        left.binding_logical_elements[binding] !=
            right.binding_logical_elements[binding] ||
        left.binding_logical_strides[binding] !=
            right.binding_logical_strides[binding] ||
        left.binding_logical_counts[binding] !=
            right.binding_logical_counts[binding]) {
      return false;
    }
  }
  return true;
}

bool same_frame(const VulkanResidencyGraphStageGeneratedProof &left,
                const VulkanResidencyGraphStageGeneratedProof &right) noexcept {
  if (!same_semantics(left, right) || left.count_id != right.count_id ||
      left.predicate_id != right.predicate_id ||
      left.count_bytes != right.count_bytes ||
      left.predicate_bytes != right.predicate_bytes ||
      left.count_handle.get() != right.count_handle.get() ||
      left.predicate_handle.get() != right.predicate_handle.get() ||
      left.frame_index != right.frame_index ||
      left.gate_generation != right.gate_generation ||
      left.valid != right.valid) {
    return false;
  }
  for (std::size_t binding = 0u; binding < left.binding_count; ++binding) {
    if (left.binding_ids[binding] != right.binding_ids[binding] ||
        left.binding_bytes[binding] != right.binding_bytes[binding] ||
        left.binding_offsets[binding] != right.binding_offsets[binding] ||
        left.binding_elements[binding] != right.binding_elements[binding] ||
        left.binding_strides[binding] != right.binding_strides[binding] ||
        left.binding_counts[binding] != right.binding_counts[binding] ||
        left.binding_handles[binding].get() !=
            right.binding_handles[binding].get()) {
      return false;
    }
  }
  return true;
}

bool graph_proof_matches(const VulkanPipeline &pipeline,
                         const VulkanResidencySelection &selection) noexcept {
  VulkanResidencyGraphStageGeneratedProof current{};
  if (!selection.graph_generated.valid || !eligible_graph(pipeline, current)) {
    return false;
  }
  if (!same_semantics(current, selection.graph_generated)) {
    return false;
  }
  if (selection.graph_generated.local_count !=
      pipeline.record->entries.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < selection.graph_generated.local_count;
       ++index) {
    VulkanResidencyGraphStageGeneratedProof frame{};
    std::uint64_t expected_generation{};
    if (!selection.graph_generated_frames[index].ready ||
        !eligible_entry(pipeline, index, frame) ||
        !rund::kernel::checked::add(static_cast<std::uint64_t>(index), 1u,
                                    expected_generation) ||
        selection.graph_generated_frames[index].generation !=
            expected_generation ||
        selection.graph_generated_frames[index]
                .expected_descriptor_generation != expected_generation) {
      return false;
    }
    frame.gate_generation = expected_generation;
    if (!same_frame(frame, selection.graph_generated_frames[index].proof)) {
      return false;
    }
  }
  return true;
}

bool sequence_proof_matches(
    const VulkanPipeline &pipeline,
    const VulkanResidencySelection &selection) noexcept {
  VulkanResidencyGraphStageSequenceProof current{};
  if (!selection.graph_sequence.valid ||
      !eligible_sequence(pipeline, current) ||
      current.stage_count != selection.graph_sequence.stage_count ||
      current.local_count != selection.graph_sequence.local_count) {
    return false;
  }
  for (std::size_t step = 0u; step < 2u; ++step) {
    auto stored = selection.graph_sequence.steps[step];
    current.steps[step].gate_generation = 0u;
    stored.gate_generation = 0u;
    if (!same_frame(current.steps[step], stored)) {
      return false;
    }
  }
  for (std::size_t local = 0u; local < selection.graph_sequence.local_count;
       ++local) {
    const auto &frame = selection.graph_sequence_frames[local];
    if (!frame.command_ready || frame.command.buffer == VK_NULL_HANDLE ||
        frame.quarantined || local >= PreparedPipelineStepCapacity) {
      return false;
    }
    for (std::size_t step = 0u; step < 2u; ++step) {
      auto current_frame = current.frame_proofs[local][step];
      auto stored_frame = frame.proofs[step];
      current_frame.gate_generation = 0u;
      stored_frame.gate_generation = 0u;
      std::uint64_t expected{};
      if (!rund::kernel::checked::add(
              static_cast<std::uint64_t>(local * 2u + step), 1u, expected) ||
          !frame.ready[step] || frame.generations[step] != expected ||
          !same_frame(current_frame, stored_frame)) {
        return false;
      }
    }
  }
  return true;
}

} // namespace impl


#endif

} // namespace rund::node::accel::detail::vulkan_generated_indirect_detail
