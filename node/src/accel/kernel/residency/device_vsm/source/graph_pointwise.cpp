#include "graph_pointwise.hpp"

#include "graph_pointwise/internal.hpp"

#include "../../../../../hash/fnv.hpp"
#include "../graph_wavefront.hpp"

#include <bit>
#include <limits>
#include <new>

namespace rund::node::accel::detail {

DeviceVsmGraphPointwiseArtifact BuildDeviceVsmGraphPointwiseArtifact(
    const std::span<const DeviceVsmGraphPointwiseStage> stages,
    const DeviceVsmGraphPointwiseTopology &topology,
    const DeviceVsmPageGeometry &geometry, const DeviceVsmPageMap &page_map,
    const DeviceVsmGraphWavefrontProof &wavefront) noexcept {
  DeviceVsmGraphPointwiseArtifact result{};
  if (stages.size() < 2u || stages.size() > DeviceVsmGraphStageCapacity ||
      wavefront.stage_count != stages.size() ||
      topology.stage_count != stages.size()) {
    result.reason = "device_vsm_graph_pointwise_stage_count_invalid";
    return result;
  }
  const DeviceVsmGraphPointwiseStage &first_stage = stages.front();
  if (first_stage.artifact == nullptr || first_stage.input == nullptr ||
      first_stage.semantic == nullptr) {
    result.reason = "device_vsm_graph_pointwise_stage_invalid";
    return result;
  }
  const rund::kernel::LoweringArtifact &first = *first_stage.artifact;
  const std::uint32_t element_bytes =
      first.key.scalar == rund::kernel::ComputeScalar::Lane32
          ? sizeof(std::uint32_t)
          : first.key.scalar == rund::kernel::ComputeScalar::Lane64
                ? sizeof(std::uint64_t)
                : 0u;
  if (element_bytes == 0u ||
      !((first.key.scalar == rund::kernel::ComputeScalar::Lane32 &&
         first.key.domain == rund::kernel::ComputeDomain::U32) ||
        (first.key.scalar == rund::kernel::ComputeScalar::Lane64 &&
         first.key.domain == rund::kernel::ComputeDomain::U64)) ||
      geometry.element_bytes != element_bytes) {
    result.reason = "device_vsm_graph_pointwise_type_mismatch";
    return result;
  }
  for (std::size_t index = 0u; index < stages.size(); ++index) {
    const DeviceVsmGraphPointwiseStage &stage = stages[index];
    if (stage.artifact == nullptr || stage.input == nullptr ||
        stage.semantic == nullptr ||
        !device_vsm_graph_pointwise::validate_stage(
            *stage.artifact, *stage.input, *stage.semantic) ||
        stage.artifact->metadata.read_count !=
            topology.stages[index].input_count) {
      result.reason = "device_vsm_graph_pointwise_stage_invalid";
      return result;
    }
    if (stage.artifact->key.api != first.key.api ||
        stage.artifact->key.scalar != first.key.scalar ||
        stage.artifact->key.domain != first.key.domain) {
      result.reason = "device_vsm_graph_pointwise_type_mismatch";
      return result;
    }
  }
  if (!device_vsm_complete_frame_geometry(geometry)) {
    result.reason = "device_vsm_graph_pointwise_geometry_invalid";
    return result;
  }
  if (!device_vsm_graph_wavefront_valid(wavefront, geometry.page_count) ||
      wavefront.stage_count != stages.size() ||
      !device_vsm_graph_pointwise_topology_valid(topology, wavefront) ||
      topology.external_input_count == 0u ||
      topology.external_input_count >= DeviceVsmResidentCapacity) {
    result.reason = "device_vsm_graph_pointwise_wavefront_invalid";
    return result;
  }
  if (geometry.logical_bytes / geometry.element_bytes >
      std::numeric_limits<std::uint32_t>::max()) {
    result.reason = "device_vsm_graph_pointwise_element_count_invalid";
    return result;
  }
  if (!device_vsm_page_map_valid(page_map, geometry.page_count) ||
      (device_vsm_page_map_active(page_map) &&
       (geometry.frame_bytes == 0u ||
        wavefront.frame_capacity !=
            page_map.words[DeviceVsmPageMapHeaderWords + 2u]))) {
    result.reason = "device_vsm_graph_pointwise_page_map_invalid";
    return result;
  }
  try {
    ::rund::node::hash_detail::Fnv hi{
        ::rund::node::hash_detail::kFnvStandardOffset};
    ::rund::node::hash_detail::Fnv lo{};
    const auto mix = [&](const std::uint64_t value) noexcept {
      hi.Number(value);
      lo.Number(value ^ 0x9e3779b97f4a7c15ull);
    };
    for (const DeviceVsmGraphPointwiseStage &stage : stages) {
      mix(stage.artifact->key.op_hash_hi);
      mix(stage.artifact->key.op_hash_lo);
      mix(stage.artifact->key.canonical_ir_hash_hi);
      mix(stage.artifact->key.canonical_ir_hash_lo);
    }
    mix(topology.stage_count);
    mix(topology.external_input_count);
    for (std::size_t stage = 0u; stage < topology.stage_count; ++stage) {
      mix(topology.stages[stage].input_count);
      for (std::size_t input = 0u; input < topology.stages[stage].input_count;
           ++input) {
        mix(static_cast<std::uint8_t>(
            topology.stages[stage].inputs[input].kind));
        mix(topology.stages[stage].inputs[input].index);
      }
    }
    mix(wavefront.stage_count);
    mix(wavefront.frame_capacity);
    mix(wavefront.batch_count);
    for (std::size_t stage = 0u; stage < DeviceVsmGraphStageCapacity; ++stage) {
      mix(wavefront.same_dispatch[stage]);
      mix(wavefront.same_release[stage]);
      mix(wavefront.prior_dispatch[stage]);
      mix(wavefront.prior_release[stage]);
    }
    if (device_vsm_page_map_active(page_map)) {
      mix(DeviceVsmPageMapMagic);
      mix(DeviceVsmPageMapVersion);
      for (const std::uint32_t word : page_map.words) {
        mix(word);
      }
    }
    rund::kernel::ArtifactKey key = first.key;
    key.variant = rund::kernel::LoweringArtifactVariant::DeviceVsm;
    key.op_hash_hi = hi.Finish();
    key.op_hash_lo = lo.Finish();
    key.canonical_ir_hash_hi = hi.Finish();
    key.canonical_ir_hash_lo = lo.Finish();
    std::string text;
    if (key.api == rund::kernel::ComputeApi::Metal) {
      text = device_vsm_graph_pointwise::metal_source(
          key, stages, topology, geometry, page_map, wavefront);
    } else if (key.api == rund::kernel::ComputeApi::Vulkan) {
      text = device_vsm_graph_pointwise::vulkan_source(
          key, stages, topology, geometry, page_map, wavefront);
    }
    if (text.empty()) {
      result.reason = "device_vsm_graph_pointwise_backend_invalid";
      return result;
    }
    result.artifact = first;
    result.artifact.key = key;
    result.artifact.kind =
        key.api == rund::kernel::ComputeApi::Metal
            ? rund::kernel::LoweringArtifactKind::MetalSource
            : rund::kernel::LoweringArtifactKind::VulkanSource;
    result.artifact.source_text = std::move(text);
    result.artifact.source_text_upper_bytes =
        result.artifact.source_text.size();
    result.artifact.ok = true;
    result.artifact.reason = "ok";
    result.plan = rund::kernel::ComputePlan{
        .tile_count = geometry.logical_bytes / geometry.element_bytes,
        .op_hash_hi = key.op_hash_hi,
        .op_hash_lo = key.op_hash_lo,
        .api = key.api,
        .scalar = first.key.scalar,
        .domain = first.key.domain,
        .input_buffer_count = topology.external_input_count,
        .output_buffer_count = 1u,
        .input_bytes_per_tile =
            topology.external_input_count * geometry.element_bytes,
        .output_bytes_per_tile = geometry.element_bytes,
        .bytes_per_tile =
            (topology.external_input_count + 1u) * geometry.element_bytes,
        .dispatch_window_tiles =
            geometry.logical_bytes / geometry.element_bytes,
        .dispatch_count = 1u,
        .ok = true,
        .reason = "ok",
    };
    result.proof = DeviceVsmGraphPointwiseProof{
        .wavefront = wavefront,
        .topology = topology,
        .page_map = page_map,
        .workgroup_width = 256u,
        .stage_count = static_cast<std::uint32_t>(stages.size())};
    result.reason = "ok";
    return result;
  } catch (const std::bad_alloc &) {
    result.reason = "compute_pipeline_capacity";
    return result;
  } catch (const std::length_error &) {
    result.reason = "compute_pipeline_capacity";
    return result;
  }
}

} // namespace rund::node::accel::detail
