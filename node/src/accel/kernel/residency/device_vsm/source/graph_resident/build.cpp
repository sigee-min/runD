#include "internal.hpp"

#include <algorithm>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>

namespace rund::node::accel::detail {

DeviceVsmGraphResidentArtifact BuildDeviceVsmGraphResidentArtifact(
    const std::span<const DeviceVsmGraphResidentStageSource> stages,
    const DeviceVsmGraphResidentProof &proof,
    const DeviceVsmGraphWavefrontProof &wavefront,
    const DeviceVsmPageGeometry &geometry,
    const DeviceVsmResidentSet &residents) noexcept {
  DeviceVsmGraphResidentArtifact result{};
  if (stages.size() < 2u || stages.size() != proof.stage_count ||
      !device_vsm_graph_resident_type_valid(proof.type) ||
      geometry.element_bytes != proof.type.element_bytes ||
      geometry.element_bytes == 0u || geometry.logical_bytes == 0u ||
      geometry.logical_bytes % geometry.element_bytes != 0u ||
      geometry.logical_bytes / geometry.element_bytes >
          std::numeric_limits<std::uint32_t>::max() ||
      geometry.payload_bytes == 0u ||
      geometry.payload_bytes % geometry.element_bytes != 0u ||
      !device_vsm_graph_resident_valid(
          proof, geometry.payload_bytes / geometry.element_bytes,
          residents.input_count, geometry.page_count) ||
      !device_vsm_graph_wavefront_valid(wavefront, geometry.page_count) ||
      wavefront.stage_count != stages.size() ||
      !graph_resident_endpoints_valid(geometry, residents)) {
    result.reason = "device_vsm_graph_resident_admission_invalid";
    return result;
  }
  DeviceVsmGraphTileSchedule tile_schedule{};
  if (!device_vsm_graph_tile_schedule(wavefront, tile_schedule) ||
      tile_schedule.count == 0u) {
    result.reason = "device_vsm_graph_resident_schedule_invalid";
    return result;
  }
  try {
    const char *stage_reason = nullptr;
    if (!graph_resident_stages_valid(stages, proof, stage_reason)) {
      result.reason = stage_reason == nullptr
                          ? "device_vsm_graph_resident_stage_invalid"
                          : stage_reason;
      return result;
    }
    rund::kernel::ArtifactKey key{};
    if (!device_vsm_graph_resident_source::detail::graph_resident_key(
            stages, proof, wavefront, key)) {
      result.reason = "device_vsm_graph_resident_stage_invalid";
      return result;
    }
    result.artifact = *stages.front().artifact;
    result.artifact.key = key;
    result.artifact.kind =
        key.api == rund::kernel::ComputeApi::Metal
            ? rund::kernel::LoweringArtifactKind::MetalSource
            : rund::kernel::LoweringArtifactKind::VulkanSource;
    result.artifact.source_text =
        key.api == rund::kernel::ComputeApi::Metal
            ? device_vsm_graph_resident_source::detail::metal_source(
                  key, stages, proof, wavefront)
            : device_vsm_graph_resident_source::detail::vulkan_source(
                  key, stages, proof, wavefront);
    if (result.artifact.source_text.empty()) {
      result.reason = "device_vsm_graph_resident_source_invalid";
      return result;
    }
    result.artifact.source_text_upper_bytes =
        result.artifact.source_text.size();
    result.artifact.ok = true;
    result.artifact.reason = "ok";
    const std::uint64_t elements =
        geometry.logical_bytes / geometry.element_bytes;
    result.plan = rund::kernel::ComputePlan{
        .tile_count = elements,
        .op_hash_hi = key.op_hash_hi,
        .op_hash_lo = key.op_hash_lo,
        .api = key.api,
        .scalar = proof.type.scalar,
        .domain = proof.type.domain,
        .input_buffer_count = residents.input_count,
        .output_buffer_count = 1u,
        .input_bytes_per_tile = residents.input_count * geometry.element_bytes,
        .output_bytes_per_tile = geometry.element_bytes,
        .bytes_per_tile = (residents.input_count + 1u) * geometry.element_bytes,
        .dispatch_window_tiles = elements,
        .dispatch_count = 1u,
        .ok = true,
        .reason = "ok",
    };
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
