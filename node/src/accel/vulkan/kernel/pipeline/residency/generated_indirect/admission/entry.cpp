#include "match.hpp"

#include <kernel/program/compute/backend.hpp>

#include <cstddef>
#include <cstdint>

namespace rund::node::accel::detail::vulkan_generated_indirect_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)


namespace impl {

[[nodiscard]] bool
eligible_entry(const VulkanPipeline &pipeline, const std::size_t index,
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
  if (pipeline.record == nullptr || index >= pipeline.record->entries.size() ||
      index >= PreparedPipelineStepCapacity) {
    return reject(VulkanResidencyGraphGeneratedPredicate::Pipeline,
                  static_cast<std::uint32_t>(index),
                  "graph_generated_pipeline");
  }
  const VulkanPipelineRecordEntry &record_entry =
      pipeline.record->entries[index];
  if (record_entry.template_index != index ||
      record_entry.occurrence_index != index ||
      record_entry.window.has_value() ||
      record_entry.transducer != NoTileTransducer) {
    return reject(VulkanResidencyGraphGeneratedPredicate::EntryIdentity,
                  static_cast<std::uint32_t>(index),
                  "graph_generated_entry_identity");
  }
  const auto *const resources =
      record_entry.prepared == nullptr
          ? nullptr
          : static_cast<const VulkanKernelResources *>(
                record_entry.prepared.get());
  const BackendRun *const run = pipeline_run(record_entry);
  const BoundStep *const bound = entry_step(record_entry, run);
  const VulkanKernelEntry *const kernel =
      resources == nullptr ? nullptr : resources->entry(0u);
  const auto *const map = kernel == nullptr
                              ? nullptr
                              : static_cast<const VulkanMapEncodeResources *>(
                                    kernel->resource.get());
  const KernelExecution *const execution =
      run == nullptr ? nullptr : run->execution;
  if (resources == nullptr || resources->size() != 1u || run == nullptr ||
      execution == nullptr || bound == nullptr || map == nullptr ||
      map->prepared == nullptr || bound->step == nullptr ||
      bound->source_binds == nullptr ||
      bound->step->kind() != rund::kernel::NodeKind::Map ||
      !bound->step->graph_binding_indices_ok ||
      !bound->step->graph_binding_indices.valid() ||
      map->prepared->checks.size() != 0u ||
      map->prepared->plan.scalar != rund::kernel::ComputeScalar::Lane64 ||
      map->prepared->plan.domain != rund::kernel::ComputeDomain::U64 ||
      !execution->admission.check.ok ||
      execution->admission.graph_id_hi == 0u ||
      execution->admission.graph_id_lo == 0u ||
      execution->admission.node_count == 0u ||
      execution->steps.size() != run->step_count || run->steps == nullptr ||
      !map->control.valid(bound->source_binds->size()) ||
      (map->control.has_count() &&
       (!map->control_count.check.ok || map->control_count.handle == nullptr ||
        map->control_count.device_buffer == nullptr)) ||
      (map->control.has_predicate() &&
       (!map->control_predicate.check.ok ||
        map->control_predicate.handle == nullptr ||
        map->control_predicate.device_buffer == nullptr))) {
    return reject(VulkanResidencyGraphGeneratedPredicate::EntryShape,
                  static_cast<std::uint32_t>(index),
                  "graph_generated_entry_shape");
  }
  const auto &accesses = bound->step->artifact.metadata.binding_accesses;
  const std::size_t data_binding_count = accesses.size();
  const std::size_t control_binding_count =
      static_cast<std::size_t>(map->control.has_count()) +
      static_cast<std::size_t>(map->control.has_predicate());
  if (data_binding_count < 2u ||
      data_binding_count > VulkanResidencyGraphStageGeneratedProof::
                               GraphGeneratedDataBindingCapacity ||
      bound->step->graph_binding_indices.size() !=
          data_binding_count + control_binding_count ||
      map->bindings.resident_inputs.count == 0u ||
      map->bindings.resident_inputs.count > 7u ||
      map->bindings.resident_outputs.count != 1u ||
      bound->source_binds->size() != execution->graph_roles.size() ||
      execution->graph_shapes.size() != execution->graph_roles.size() ||
      execution->graph_alias_representatives.size() !=
          execution->graph_roles.size()) {
    return reject(VulkanResidencyGraphGeneratedPredicate::EntryShape,
                  static_cast<std::uint32_t>(index),
                  "graph_generated_binding_shape");
  }
  std::size_t read_position = 0u;
  std::size_t write_position = 0u;
  for (std::size_t position = 0u; position < accesses.size(); ++position) {
    const std::uint64_t graph_binding =
        bound->step->graph_binding_indices[position];
    if (graph_binding >= bound->source_binds->size() ||
        graph_binding >= execution->graph_roles.size()) {
      return reject(VulkanResidencyGraphGeneratedPredicate::BindingIndex,
                    static_cast<std::uint32_t>(position),
                    "graph_generated_binding_index");
    }
    const bool read =
        accesses[position] == rund::kernel::ComputeBindingAccess::Read;
    const bool write =
        accesses[position] == rund::kernel::ComputeBindingAccess::Write;
    if (!read && !write) {
      return reject(VulkanResidencyGraphGeneratedPredicate::BindingRole,
                    static_cast<std::uint32_t>(position),
                    "graph_generated_binding_role");
    }
    const std::size_t data_position = read ? read_position++ : write_position++;
    const rund::kernel::ResidentBindingRange range =
        read ? map->bindings.resident_inputs : map->bindings.resident_outputs;
    if (data_position >= range.count ||
        !copy_binding(range.ref(data_position), range.handle(data_position),
                      position, proof) ||
        !graph_alias_matches(*execution, *bound->source_binds, graph_binding,
                             *range.ref(data_position),
                             range.handle(data_position))) {
      return reject(VulkanResidencyGraphGeneratedPredicate::BindingAlias,
                    static_cast<std::uint32_t>(position),
                    "graph_generated_binding_alias");
    }
    proof.binding_roles[position] =
        static_cast<std::uint8_t>(accesses[position]);
    proof.binding_logical_elements[position] =
        range.ref(data_position)->element_bytes;
    proof.binding_logical_strides[position] =
        range.ref(data_position)->stride_bytes;
    proof.binding_logical_counts[position] = range.ref(data_position)->count;
    if ((read && execution->graph_roles[graph_binding] !=
                     rund::kernel::BufferRole::Read) ||
        (write && execution->graph_roles[graph_binding] !=
                      rund::kernel::BufferRole::Write)) {
      return reject(VulkanResidencyGraphGeneratedPredicate::BindingRole,
                    static_cast<std::uint32_t>(position),
                    "graph_generated_graph_role");
    }
  }
  if (read_position == 0u || read_position > 7u || write_position != 1u) {
    return reject(VulkanResidencyGraphGeneratedPredicate::EntryShape,
                  static_cast<std::uint32_t>(index),
                  "graph_generated_access_cardinality");
  }
  const auto check_control = [&](const std::size_t slot,
                                 const std::uint32_t binding,
                                 const rund::kernel::GraphControlSource source,
                                 const std::uint64_t offset,
                                 const VulkanResidentBufferResult &resident) {
    const std::uint64_t element_bytes =
        source == rund::kernel::GraphControlSource::U32
            ? 4u
            : (source == rund::kernel::GraphControlSource::U64 ? 8u : 0u);
    if (binding != slot || slot < data_binding_count ||
        slot >= bound->step->graph_binding_indices.size() ||
        element_bytes == 0u || resident.handle == nullptr ||
        resident.device_buffer == nullptr ||
        resident.ref.element_bytes != element_bytes ||
        offset > resident.ref.bytes ||
        element_bytes > resident.ref.bytes - offset ||
        (offset % element_bytes) != 0u) {
      return false;
    }
    const std::uint64_t graph_binding =
        bound->step->graph_binding_indices[slot];
    if (graph_binding >= execution->graph_roles.size() ||
        execution->graph_roles[graph_binding] !=
            rund::kernel::BufferRole::Read) {
      return false;
    }
    return graph_alias_matches(*execution, *bound->source_binds, graph_binding,
                               resident.ref, &resident.handle);
  };
  if (map->control.has_count() &&
      !check_control(data_binding_count, map->control.count_binding,
                     map->control.count_source, map->control.count_byte_offset,
                     map->control_count)) {
    return reject(VulkanResidencyGraphGeneratedPredicate::ControlBinding,
                  map->control.count_binding, "graph_generated_count_control");
  }
  if (map->control.has_predicate() &&
      !check_control(
          data_binding_count +
              static_cast<std::size_t>(map->control.has_count()),
          map->control.predicate_binding, map->control.predicate_source,
          map->control.predicate_byte_offset, map->control_predicate)) {
    return reject(VulkanResidencyGraphGeneratedPredicate::ControlBinding,
                  map->control.predicate_binding,
                  "graph_generated_predicate_control");
  }
  proof.kernel_id = execution->admission.kernel_id;
  proof.graph_id_hi = execution->admission.graph_id_hi;
  proof.graph_id_lo = execution->admission.graph_id_lo;
  proof.node_count = execution->admission.node_count;
  proof.op_hash_hi = map->prepared->plan.op_hash_hi;
  proof.op_hash_lo = map->prepared->plan.op_hash_lo;
  proof.binding_count = static_cast<std::uint32_t>(accesses.size());
  proof.read_count = static_cast<std::uint32_t>(read_position);
  proof.write_count = static_cast<std::uint32_t>(write_position);
  proof.local_count =
      static_cast<std::uint32_t>(pipeline.record->entries.size());
  proof.frame_index = static_cast<std::uint32_t>(index);
  proof.count_binding = map->control.count_binding;
  proof.predicate_binding = map->control.predicate_binding;
  proof.count_offset = map->control.count_byte_offset;
  proof.predicate_offset = map->control.predicate_byte_offset;
  proof.count_capacity = map->control.capacity;
  proof.predicate_expected = map->control.predicate_expected;
  proof.count_id = map->control_count.ref.id;
  proof.count_bytes = map->control_count.ref.bytes;
  proof.count_handle = map->control_count.handle;
  if (map->control.has_predicate()) {
    proof.predicate_id = map->control_predicate.ref.id;
    proof.predicate_bytes = map->control_predicate.ref.bytes;
    proof.predicate_handle = map->control_predicate.handle;
  }
  proof.count_source = static_cast<std::uint8_t>(map->control.count_source);
  proof.predicate_source =
      static_cast<std::uint8_t>(map->control.predicate_source);
  proof.has_count = map->control.has_count();
  proof.has_predicate = map->control.has_predicate();
  proof.valid = true;
  return true;
}


bool eligible_graph(const VulkanPipeline &pipeline,
                    VulkanResidencyGraphStageGeneratedProof &proof) noexcept {
  proof = {};
  if (pipeline.record == nullptr || pipeline.record->entries.empty() ||
      pipeline.record->entries.size() > PreparedPipelineStepCapacity ||
      pipeline.record->recurrence || pipeline.profile != nullptr ||
      !pipeline.transducers.empty() || !pipeline.window.routes.empty() ||
      !pipeline.window.gates.empty() || !pipeline.publish.routes.empty()) {
    proof.first_failure_predicate =
        VulkanResidencyGraphGeneratedPredicate::Pipeline;
    proof.first_failure_reason = "graph_generated_pipeline";
    return false;
  }
  const std::size_t count = pipeline.record->entries.size();
  for (std::size_t index = 0u; index < count; ++index) {
    VulkanResidencyGraphStageGeneratedProof current{};
    if (!eligible_entry(pipeline, index, current)) {
      proof.first_failure_predicate = current.first_failure_predicate;
      proof.first_failure_ordinal = current.first_failure_ordinal;
      proof.first_failure_reason = current.first_failure_reason;
      return false;
    }
    if (index == 0u) {
      proof = current;
    } else if (!same_semantics(current, proof)) {
      proof.first_failure_predicate =
          VulkanResidencyGraphGeneratedPredicate::Semantic;
      proof.first_failure_ordinal = static_cast<std::uint32_t>(index);
      proof.first_failure_reason = "graph_generated_semantic";
      return false;
    }
  }
  proof.local_count = static_cast<std::uint32_t>(count);
  proof.valid = true;
  return true;
}

} // namespace impl


#endif

} // namespace rund::node::accel::detail::vulkan_generated_indirect_detail
