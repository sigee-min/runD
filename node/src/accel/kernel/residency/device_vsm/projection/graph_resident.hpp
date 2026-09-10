#pragma once

#include "../projection.hpp"

#include "../../../prepared/model.hpp"

#include <kernel/core/checked.hpp>

#include <array>
#include <limits>

namespace rund::node::accel::detail::device_vsm_graph_projection {

using CheckPipeline = bool (*)(const prepared::PipelineState &,
                               const BoundStep *&, rund::kernel::BindingSet &,
                               const char *&);

struct GraphPick final {
  DeviceVsmGraphResidentProof proof{};
  std::array<PreparedKernelPipeline, DeviceVsmGraphResidentStageCapacity>
      pipelines{};
  std::array<const prepared::PipelineState *,
             DeviceVsmGraphResidentStageCapacity>
      states{};
  std::array<const BoundStep *, DeviceVsmGraphResidentStageCapacity> steps{};
  std::array<rund::kernel::BindingSet, DeviceVsmGraphResidentStageCapacity>
      bindings{};
  const char *reason{"device_vsm_graph_resident_stage_invalid"};
  bool requested{};
  bool ready{};
};

[[nodiscard]] inline bool graph_resident_geometry_witness(
    const DeviceVsmGraphResidentProof &proof,
    const DeviceVsmPageGeometry &geometry,
    const std::uint64_t external_input_count,
    std::uint64_t &page_elements) noexcept {
  const DeviceVsmGraphResidentType &type = proof.type;
  if (!device_vsm_graph_resident_type_valid(type) ||
      proof.resource_count == 0u || type.element_bytes == 0u ||
      geometry.logical_bytes == 0u || geometry.payload_bytes == 0u ||
      geometry.frame_bytes != geometry.payload_bytes ||
      geometry.element_bytes != type.element_bytes ||
      geometry.logical_bytes % type.element_bytes != 0u ||
      geometry.payload_bytes % type.element_bytes != 0u ||
      proof.resources[0u].page_bytes != geometry.payload_bytes ||
      external_input_count >=
          std::numeric_limits<std::uint64_t>::max() - 1u) {
    return false;
  }
  page_elements = geometry.payload_bytes / type.element_bytes;
  if (page_elements == 0u) {
    return false;
  }
  std::uint64_t external_count = 0u;
  for (std::size_t index = 0u; index < proof.resource_count; ++index) {
    const DeviceVsmGraphResidentResource &resource = proof.resources[index];
    if (resource.external_slot == DeviceVsmGraphResidentExternalSlotInvalid) {
      continue;
    }
    if (resource.external_slot > external_input_count ||
        resource.page_bytes != geometry.payload_bytes ||
        resource.logical_bytes != geometry.logical_bytes ||
        resource.type != device_vsm_graph_resident_type_code(type)) {
      return false;
    }
    ++external_count;
  }
  return external_count == external_input_count + 1u;
}

[[nodiscard]] inline bool graph_resident_ref_witness(
    const rund::kernel::ResidentBindingRange &range,
    const std::uint64_t expected_count, const std::uint64_t page_elements,
    const std::uint64_t page_bytes, const std::uint32_t element_bytes,
    const std::uint32_t usage) noexcept {
  if (range.count != expected_count || !range.has_refs()) {
    return false;
  }
  for (std::uint64_t index = 0u; index < range.count; ++index) {
    const rund::kernel::ResidentBufferRef *const ref = range.ref(index);
    const std::shared_ptr<void> *const handle = range.handle(index);
    std::uint64_t tail = 0u;
    std::uint64_t extent = 0u;
    if (ref == nullptr || handle == nullptr || *handle == nullptr ||
        ref->id == 0u || ref->bytes < page_bytes ||
        ref->offset_bytes != 0u || ref->element_bytes != element_bytes ||
        ref->stride_bytes != element_bytes || ref->count < page_elements ||
        ref->count == 0u || ref->usage != usage ||
        !::rund::kernel::checked::mul(ref->count - 1u, ref->stride_bytes,
                                     tail) ||
        !::rund::kernel::checked::add(tail, ref->element_bytes, extent) ||
        extent > ref->bytes) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline bool graph_resident_stage_geometry_witness(
    const BoundStep &bound, const rund::kernel::BindingSet &bindings,
    const DeviceVsmGraphResidentType &type,
    const DeviceVsmPageGeometry &geometry, const std::uint64_t page_elements,
    const std::uint64_t input_count) noexcept {
  if (bound.step == nullptr || bound.planned == nullptr ||
      !bound.step->artifact.ok || !bound.step->artifact.metadata.ok ||
      !device_vsm_graph_resident_type_valid(type) || page_elements == 0u ||
      type.element_bytes == 0u) {
    return false;
  }
  const rund::kernel::LoweringArtifact &artifact = bound.step->artifact;
  const rund::kernel::ExecutionMetadata &metadata = artifact.metadata;
  const rund::kernel::ComputeMap &map = metadata.map;
  const rund::kernel::ComputePlan &plan = bound.planned->plan;
  const rund::kernel::ArtifactKey &key = artifact.key;
  std::uint64_t input_bytes = 0u;
  if (!::rund::kernel::checked::mul(input_count, type.element_bytes,
                                   input_bytes)) {
    return false;
  }
  if ((key.api != rund::kernel::ComputeApi::Metal &&
       key.api != rund::kernel::ComputeApi::Vulkan) ||
      key.scalar != type.scalar || key.domain != type.domain ||
      !rund::kernel::ComputeFixedFormatAbsent(key.fixed_format) ||
      bound.planned->domain != type.domain ||
      map.api != key.api || map.scalar != type.scalar ||
      map.domain != type.domain || map.fixed_format != key.fixed_format ||
      map.input_buffer_count != input_count ||
      map.output_buffer_count != 1u ||
      map.input_bytes_per_tile != input_bytes ||
      map.output_bytes_per_tile != type.element_bytes || map.param_bytes != 0u ||
      map.metadata_bytes_per_tile != sizeof(std::uint32_t) ||
      metadata.read_count != input_count || metadata.write_count != 1u ||
      metadata.input_element_bytes.size() != input_count ||
      metadata.output_element_bytes.size() != 1u ||
      metadata.param_storage.size() != map.param_bytes) {
    return false;
  }
  for (const std::uint64_t width : metadata.input_element_bytes) {
    if (width != type.element_bytes) {
      return false;
    }
  }
  if (metadata.output_element_bytes.front() != type.element_bytes) {
    return false;
  }
  if (!plan.ok || !plan.fixed_authoritative || plan.api != key.api ||
      plan.scalar != type.scalar || plan.domain != type.domain ||
      plan.fixed_format != key.fixed_format || plan.tile_count != page_elements ||
      plan.input_buffer_count != input_count || plan.output_buffer_count != 1u ||
      plan.input_bytes_per_tile != input_bytes ||
      plan.output_bytes_per_tile != type.element_bytes ||
      plan.param_bytes != map.param_bytes ||
      plan.metadata_bytes_per_tile != map.metadata_bytes_per_tile ||
      plan.dispatch_window_tiles == 0u ||
      plan.dispatch_window_tiles > page_elements ||
      plan.dispatch_count !=
          page_elements / plan.dispatch_window_tiles +
              static_cast<std::uint64_t>(
                  page_elements % plan.dispatch_window_tiles != 0u)) {
    return false;
  }
  std::uint64_t bytes_per_tile = 0u;
  std::uint64_t staging_bytes = 0u;
  if (!::rund::kernel::checked::add(input_bytes, type.element_bytes,
                              bytes_per_tile) ||
      !::rund::kernel::checked::add(bytes_per_tile, plan.metadata_bytes_per_tile,
                              bytes_per_tile) ||
      bytes_per_tile != plan.bytes_per_tile ||
      !::rund::kernel::checked::mul(plan.bytes_per_tile,
                                   plan.dispatch_window_tiles, staging_bytes) ||
      !::rund::kernel::checked::add(staging_bytes, plan.param_bytes, staging_bytes) ||
      staging_bytes != plan.staging_bytes) {
    return false;
  }
  if (!bindings.ok || bindings.api != key.api ||
      bindings.scalar != type.scalar || bindings.domain != type.domain ||
      bindings.tile_count != page_elements || bindings.lane_count != page_elements ||
      bindings.input_bytes_per_tile != plan.input_bytes_per_tile ||
      bindings.output_bytes_per_tile != plan.output_bytes_per_tile ||
      bindings.metadata_bytes_per_tile != plan.metadata_bytes_per_tile ||
      bindings.param_bytes != 0u || bindings.param_data_bytes != 0u ||
      bindings.param_data != nullptr ||
      bindings.input_element_byte_count != input_count ||
      bindings.input_element_bytes == nullptr ||
      !graph_resident_ref_witness(
          bindings.resident_inputs, input_count, page_elements,
          geometry.payload_bytes, type.element_bytes,
          rund::kernel::kResidentUsageRead) ||
      !graph_resident_ref_witness(
          bindings.resident_outputs, 1u, page_elements, geometry.payload_bytes,
          type.element_bytes, rund::kernel::kResidentUsageWrite)) {
    return false;
  }
  for (std::uint64_t index = 0u; index < input_count; ++index) {
    if (bindings.input_element_bytes[index] != type.element_bytes) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline GraphPick pick_graph(const DeviceVsmProofRequest &request,
                                          const CheckPipeline check) noexcept {
  GraphPick picked{};
  picked.proof = request.graph_resident;
  picked.requested = request.kind == DeviceVsmProjectionKind::GraphResident;
  if (!picked.requested) {
    return picked;
  }
  if (!device_vsm_graph_resident_valid(picked.proof, 0u,
                                       request.residents.input_count,
                                       request.geometry.page_count)) {
    picked.reason = "device_vsm_graph_resident_proof_invalid";
    return picked;
  }
  std::uint64_t page_elements = 0u;
  if (!graph_resident_geometry_witness(
          picked.proof, request.geometry, request.residents.input_count,
          page_elements)) {
    picked.reason = "device_vsm_graph_resident_geometry_invalid";
    picked.ready = false;
    return picked;
  }
  picked.ready =
      request.graph_resident_pipeline_count >= 2u &&
      request.graph_resident_pipeline_count <=
          DeviceVsmGraphResidentStageCapacity &&
      request.graph_resident_pipeline_count == picked.proof.stage_count &&
      request.graph_resident_pipeline_count ==
          request.graph_wavefront.stage_count &&
      device_vsm_graph_wavefront_valid(request.graph_wavefront,
                                       request.geometry.page_count);
  if (!picked.ready) {
    return picked;
  }
  if (request.peer_pipeline.ok) {
    picked.reason = "device_vsm_graph_resident_peer_invalid";
    picked.ready = false;
    return picked;
  }
  for (std::size_t stage = 0u;
       picked.ready && stage < request.graph_resident_pipeline_count; ++stage) {
    picked.pipelines[stage] = request.graph_resident_pipelines[stage];
    const PreparedKernelPipeline &prepared = picked.pipelines[stage];
    picked.states[stage] =
        prepared.ok
            ? static_cast<const prepared::PipelineState *>(prepared.owner.get())
            : nullptr;
    picked.ready = picked.states[stage] != nullptr && check != nullptr &&
                   check(*picked.states[stage], picked.steps[stage],
                         picked.bindings[stage], picked.reason);
    if (picked.ready && (picked.steps[stage] == nullptr ||
                         (picked.steps[stage]->step->artifact.key.api !=
                              rund::kernel::ComputeApi::Vulkan &&
                          picked.steps[stage]->step->artifact.key.api !=
                              rund::kernel::ComputeApi::Metal) ||
                         picked.steps[stage]->step->artifact.key.scalar !=
                             picked.proof.type.scalar ||
                         picked.steps[stage]->step->artifact.key.domain !=
                             picked.proof.type.domain ||
                         picked.bindings[stage].param_bytes != 0u)) {
      picked.reason = "device_vsm_graph_resident_stage_invalid";
      picked.ready = false;
    }
    std::size_t reads = 0u;
    if (picked.ready) {
      for (std::size_t port = 0u; port < picked.proof.stages[stage].port_count;
           ++port) {
        reads += picked.proof.stages[stage].ports[port].access == 0u;
      }
      picked.ready =
          reads == picked.steps[stage]->step->artifact.metadata.read_count &&
          picked.steps[stage]->step->artifact.metadata.write_count == 1u;
      if (!picked.ready) {
        picked.reason = "device_vsm_graph_resident_stage_invalid";
      }
    }
    if (picked.ready && !graph_resident_stage_geometry_witness(
                            *picked.steps[stage], picked.bindings[stage],
                            picked.proof.type, request.geometry, page_elements,
                            static_cast<std::uint64_t>(reads))) {
      picked.reason = "device_vsm_graph_resident_geometry_invalid";
      picked.ready = false;
    }
  }
  if (picked.ready) {
    const auto &first = picked.steps[0u]->planned->plan;
    if (request.residents.output_count != 1u ||
        first.scalar != picked.proof.type.scalar ||
        first.domain != picked.proof.type.domain ||
        first.tile_count != page_elements ||
        request.output_bytes != request.geometry.logical_bytes ||
        first.output_buffer_count != 1u || first.param_bytes != 0u) {
      picked.reason = "device_vsm_graph_resident_geometry_invalid";
      picked.ready = false;
    }
  }
  return picked;
}

} // namespace rund::node::accel::detail::device_vsm_graph_projection
