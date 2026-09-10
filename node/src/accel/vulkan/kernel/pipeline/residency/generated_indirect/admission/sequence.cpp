#include "match.hpp"

#include <kernel/program/compute/backend.hpp>

#include <cstddef>
#include <cstdint>

#include <array>

namespace rund::node::accel::detail::vulkan_generated_indirect_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)


namespace impl {

[[nodiscard]] bool
eligible_step(const VulkanPipeline &pipeline, const std::size_t local,
              const std::size_t stage, const std::size_t expected_reads,
              const std::size_t expected_writes,
              VulkanResidencyGraphStageGeneratedProof &proof) noexcept {
  proof = {};
  const auto reject =
      [&](const VulkanResidencyGraphGeneratedPredicate predicate,
          const std::uint32_t ordinal, const char *const reason) noexcept {
        proof.first_failure_predicate = predicate;
        proof.first_failure_ordinal = ordinal;
        proof.first_failure_reason = reason;
        return false;
      };
  if (pipeline.record == nullptr || local >= pipeline.record->entries.size() ||
      local >= PreparedPipelineStepCapacity || stage >= 2u) {
    return reject(VulkanResidencyGraphGeneratedPredicate::Pipeline,
                  static_cast<std::uint32_t>(local * 2u + stage),
                  "graph_sequence_pipeline");
  }
  const VulkanPipelineRecordEntry &entry = pipeline.record->entries[local];
  if (entry.template_index != local || entry.occurrence_index != local ||
      entry.window.has_value() || entry.transducer != NoTileTransducer ||
      entry.step_index != 0u) {
    return reject(VulkanResidencyGraphGeneratedPredicate::EntryIdentity,
                  static_cast<std::uint32_t>(local * 2u + stage),
                  "graph_sequence_entry_identity");
  }
  const auto *const resources =
      entry.prepared == nullptr
          ? nullptr
          : static_cast<const VulkanKernelResources *>(entry.prepared.get());
  const BackendRun *const run = pipeline_run(entry);
  const BoundStep *const bound =
      run == nullptr || run->steps == nullptr || stage >= run->step_count
          ? nullptr
          : &run->steps[stage];
  const VulkanKernelEntry *const kernel =
      resources == nullptr ? nullptr : resources->entry(stage);
  const auto *const map = kernel == nullptr
                              ? nullptr
                              : static_cast<const VulkanMapEncodeResources *>(
                                    kernel->resource.get());
  const KernelExecution *const execution =
      run == nullptr ? nullptr : run->execution;
  if (resources == nullptr || resources->size() != 2u || run == nullptr ||
      execution == nullptr || bound == nullptr || kernel == nullptr ||
      map == nullptr || map->prepared == nullptr || bound->step == nullptr ||
      bound->source_binds == nullptr ||
      bound->step->kind() != rund::kernel::NodeKind::Map ||
      !bound->step->graph_binding_indices_ok ||
      !bound->step->graph_binding_indices.valid() ||
      !map->prepared->checks.empty() || map->history_recurrence ||
      map->iterations != 1u || !map->control.has_count() ||
      map->control.has_predicate() ||
      map->prepared->plan.scalar != rund::kernel::ComputeScalar::Lane64 ||
      map->prepared->plan.domain != rund::kernel::ComputeDomain::U64 ||
      !execution->admission.check.ok ||
      execution->admission.graph_id_hi == 0u ||
      execution->admission.graph_id_lo == 0u ||
      execution->admission.node_count == 0u || run->step_count != 2u ||
      execution->steps.size() != run->step_count || run->steps == nullptr ||
      !map->control.valid(bound->source_binds->size()) ||
      map->control.count_source != rund::kernel::GraphControlSource::U64 ||
      !map->control_count.check.ok || map->control_count.handle == nullptr ||
      map->control_count.device_buffer == nullptr) {
    return reject(VulkanResidencyGraphGeneratedPredicate::EntryShape,
                  static_cast<std::uint32_t>(local * 2u + stage),
                  "graph_sequence_entry_shape");
  }
  const auto &accesses = bound->step->artifact.metadata.binding_accesses;
  if (accesses.size() != expected_reads + expected_writes ||
      bound->step->graph_binding_indices.size() != accesses.size() + 1u ||
      map->bindings.resident_inputs.count != expected_reads ||
      map->bindings.resident_outputs.count != expected_writes ||
      bound->source_binds->size() != execution->graph_roles.size() ||
      execution->graph_shapes.size() != execution->graph_roles.size() ||
      execution->graph_alias_representatives.size() !=
          execution->graph_roles.size()) {
    return reject(VulkanResidencyGraphGeneratedPredicate::EntryShape,
                  static_cast<std::uint32_t>(local * 2u + stage),
                  "graph_sequence_binding_shape");
  }
  std::size_t reads = 0u;
  std::size_t writes = 0u;
  for (std::size_t position = 0u; position < accesses.size(); ++position) {
    const std::uint64_t graph_binding =
        bound->step->graph_binding_indices[position];
    if (graph_binding >= bound->source_binds->size() ||
        graph_binding >= execution->graph_roles.size()) {
      return reject(VulkanResidencyGraphGeneratedPredicate::BindingIndex,
                    static_cast<std::uint32_t>(local * 2u + stage),
                    "graph_sequence_binding_index");
    }
    const bool read =
        accesses[position] == rund::kernel::ComputeBindingAccess::Read;
    const bool write =
        accesses[position] == rund::kernel::ComputeBindingAccess::Write;
    if (!read && !write) {
      return reject(VulkanResidencyGraphGeneratedPredicate::BindingRole,
                    static_cast<std::uint32_t>(local * 2u + stage),
                    "graph_sequence_binding_role");
    }
    const std::size_t data_position = read ? reads++ : writes++;
    const auto range =
        read ? map->bindings.resident_inputs : map->bindings.resident_outputs;
    if (data_position >= range.count ||
        !copy_binding(range.ref(data_position), range.handle(data_position),
                      position, proof) ||
        !graph_alias_matches(*execution, *bound->source_binds, graph_binding,
                             *range.ref(data_position),
                             range.handle(data_position))) {
      return reject(VulkanResidencyGraphGeneratedPredicate::BindingAlias,
                    static_cast<std::uint32_t>(local * 2u + stage),
                    "graph_sequence_binding_alias");
    }
    proof.binding_roles[position] =
        static_cast<std::uint8_t>(accesses[position]);
    proof.binding_logical_elements[position] =
        range.ref(data_position)->element_bytes;
    proof.binding_logical_strides[position] =
        range.ref(data_position)->stride_bytes;
    proof.binding_logical_counts[position] = range.ref(data_position)->count;
    const auto expected_role =
        read ? rund::kernel::BufferRole::Read : rund::kernel::BufferRole::Write;
    if (execution->graph_roles[graph_binding] != expected_role) {
      return reject(VulkanResidencyGraphGeneratedPredicate::BindingRole,
                    static_cast<std::uint32_t>(local * 2u + stage),
                    "graph_sequence_graph_role");
    }
  }
  const std::size_t count_slot = accesses.size();
  if (reads != expected_reads || writes != expected_writes ||
      map->control.count_binding != count_slot ||
      bound->step->graph_binding_indices[count_slot] >=
          execution->graph_roles.size() ||
      execution->graph_roles[bound->step->graph_binding_indices[count_slot]] !=
          rund::kernel::BufferRole::Read ||
      map->control_count.ref.element_bytes != sizeof(std::uint64_t) ||
      map->control.count_byte_offset > map->control_count.ref.bytes ||
      sizeof(std::uint64_t) >
          map->control_count.ref.bytes - map->control.count_byte_offset ||
      (map->control.count_byte_offset % sizeof(std::uint64_t)) != 0u ||
      !graph_alias_matches(*execution, *bound->source_binds,
                           bound->step->graph_binding_indices[count_slot],
                           map->control_count.ref,
                           &map->control_count.handle)) {
    return reject(VulkanResidencyGraphGeneratedPredicate::ControlBinding,
                  static_cast<std::uint32_t>(local * 2u + stage),
                  "graph_sequence_count_control");
  }
  proof.kernel_id = execution->admission.kernel_id;
  proof.graph_id_hi = execution->admission.graph_id_hi;
  proof.graph_id_lo = execution->admission.graph_id_lo;
  proof.node_count = execution->admission.node_count;
  proof.op_hash_hi = map->prepared->plan.op_hash_hi;
  proof.op_hash_lo = map->prepared->plan.op_hash_lo;
  proof.binding_count = static_cast<std::uint32_t>(accesses.size());
  proof.read_count = static_cast<std::uint32_t>(reads);
  proof.write_count = static_cast<std::uint32_t>(writes);
  proof.local_count = 2u;
  proof.frame_index = static_cast<std::uint32_t>(local * 2u + stage);
  proof.count_binding = map->control.count_binding;
  proof.count_offset = map->control.count_byte_offset;
  proof.count_capacity = map->control.capacity;
  proof.count_id = map->control_count.ref.id;
  proof.count_bytes = map->control_count.ref.bytes;
  proof.count_handle = map->control_count.handle;
  proof.count_source = static_cast<std::uint8_t>(map->control.count_source);
  proof.has_count = true;
  proof.valid = true;
  return true;
}

bool eligible_sequence(const VulkanPipeline &pipeline,
                       VulkanResidencyGraphStageSequenceProof &proof) noexcept {
  proof = {};
  if (pipeline.record == nullptr || pipeline.record->entries.size() != 2u ||
      pipeline.record->status.generation_stride != 1u ||
      pipeline.record->recurrence || pipeline.profile != nullptr ||
      !pipeline.transducers.empty() || !pipeline.window.routes.empty() ||
      !pipeline.window.gates.empty() || !pipeline.publish.routes.empty()) {
    proof.first_failure_predicate =
        VulkanResidencyGraphGeneratedPredicate::Pipeline;
    proof.first_failure_reason =
        pipeline.record != nullptr &&
                pipeline.record->status.generation_stride != 1u
            ? "graph_sequence_generation_stride"
            : "graph_sequence_pipeline";
    return false;
  }
  for (std::size_t local = 0u; local < 2u; ++local) {
    // Sequence aliasing is meaningful only when the two logical steps own
    // distinct map/resource/step rows.  Without this identity proof, a
    // repeated owner could make a self-alias look like the stage-0 write to
    // stage-1 read handoff.
    const auto &entry = pipeline.record->entries[local];
    const auto *const resources =
        entry.prepared == nullptr
            ? nullptr
            : static_cast<const VulkanKernelResources *>(entry.prepared.get());
    const auto *const run = pipeline_run(entry);
    const auto *const first =
        resources == nullptr ? nullptr : resources->entry(0u);
    const auto *const second =
        resources == nullptr ? nullptr : resources->entry(1u);
    const auto *const first_map =
        first == nullptr ? nullptr : first->resource.get();
    const auto *const second_map =
        second == nullptr ? nullptr : second->resource.get();
    if (resources == nullptr || resources->size() != 2u || run == nullptr ||
        run->steps == nullptr || run->step_count != 2u || first == nullptr ||
        second == nullptr || first == second || first_map == nullptr ||
        second_map == nullptr || first_map == second_map ||
        &run->steps[0] == &run->steps[1]) {
      proof.first_failure_predicate =
          VulkanResidencyGraphGeneratedPredicate::EntryIdentity;
      proof.first_failure_ordinal = static_cast<std::uint32_t>(local * 2u);
      proof.first_failure_reason = "graph_sequence_step_owner_identity";
      return false;
    }
    for (std::size_t stage = 0u; stage < 2u; ++stage) {
      const std::size_t reads = stage == 0u ? 1u : 2u;
      const std::size_t writes = stage == 0u ? 2u : 1u;
      if (!eligible_step(pipeline, local, stage, reads, writes,
                         proof.frame_proofs[local][stage])) {
        proof.first_failure_predicate =
            proof.frame_proofs[local][stage].first_failure_predicate;
        proof.first_failure_ordinal =
            proof.frame_proofs[local][stage].first_failure_ordinal;
        proof.first_failure_reason =
            proof.frame_proofs[local][stage].first_failure_reason;
        return false;
      }
      if (local == 0u) {
        proof.steps[stage] = proof.frame_proofs[local][stage];
      } else if (!same_semantics(proof.steps[stage],
                                 proof.frame_proofs[local][stage])) {
        proof.first_failure_predicate =
            VulkanResidencyGraphGeneratedPredicate::Semantic;
        proof.first_failure_ordinal =
            static_cast<std::uint32_t>(local * 2u + stage);
        proof.first_failure_reason = "graph_sequence_semantic";
        return false;
      }
    }
    std::array<std::size_t, 2u> outputs{};
    std::array<std::size_t, 2u> inputs{};
    std::size_t output_count = 0u;
    std::size_t input_count = 0u;
    for (std::size_t index = 0u;
         index < proof.frame_proofs[local][0u].binding_count; ++index) {
      if (proof.frame_proofs[local][0u].binding_roles[index] ==
          static_cast<std::uint8_t>(
              rund::kernel::ComputeBindingAccess::Write)) {
        outputs[output_count++] = index;
      }
    }
    for (std::size_t index = 0u;
         index < proof.frame_proofs[local][1u].binding_count; ++index) {
      if (proof.frame_proofs[local][1u].binding_roles[index] ==
          static_cast<std::uint8_t>(rund::kernel::ComputeBindingAccess::Read)) {
        inputs[input_count++] = index;
      }
    }
    if (output_count != 2u || input_count != 2u ||
        !same_proof_storage(proof.frame_proofs[local][0u], outputs[0u],
                            proof.frame_proofs[local][1u], inputs[0u]) ||
        !same_proof_storage(proof.frame_proofs[local][0u], outputs[1u],
                            proof.frame_proofs[local][1u], inputs[1u])) {
      proof.first_failure_predicate =
          VulkanResidencyGraphGeneratedPredicate::BindingAlias;
      proof.first_failure_ordinal = static_cast<std::uint32_t>(local * 2u + 1u);
      proof.first_failure_reason = "graph_sequence_intermediate_alias";
      return false;
    }
  }
  proof.stage_count = 2u;
  proof.local_count = 2u;
  proof.valid = true;
  return true;
}

} // namespace impl


#endif

} // namespace rund::node::accel::detail::vulkan_generated_indirect_detail
