#include "graph_map_scan.hpp"

#include "graph_map_scan/internal.hpp"
#include "scan/shape.hpp"

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/scan/identity.hpp>

#include <bit>
#include <new>

namespace rund::node::accel::detail {

DeviceVsmGraphMapScanArtifact BuildDeviceVsmGraphMapScanArtifact(
    const rund::kernel::LoweringArtifact &source,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &input,
    const MapSemantic &map_semantic,
    const rund::kernel::ScanPlan &scan_semantic,
    const DeviceVsmPageGeometry &geometry) noexcept {
  DeviceVsmGraphMapScanArtifact result{};
  const device_vsm_scan_source::Shape shape =
      device_vsm_scan_source::project_shape(scan_semantic, source.key.api,
                                            geometry);
  if (!shape || !shape.u64) {
    result.reason = "device_vsm_graph_scan_geometry_invalid";
    return result;
  }
  if (!device_vsm_typed_map::validate_total_u64(source, input, map_semantic)) {
    result.reason = "device_vsm_graph_scan_ir_invalid";
    return result;
  }
  const std::uint64_t input_count = source.metadata.read_count;
  std::uint64_t input_bytes_per_tile = 0u;
  std::uint64_t bytes_per_tile = 0u;
  if (!rund::kernel::checked::mul(input_count, shape.element_bytes,
                                  input_bytes_per_tile) ||
      !rund::kernel::checked::add(input_bytes_per_tile, shape.element_bytes,
                                  bytes_per_tile)) {
    result.reason = "device_vsm_graph_scan_geometry_invalid";
    return result;
  }
  try {
    const rund::kernel::ScanDesc descriptor{
        .op = scan_semantic.op,
        .element = scan_semantic.element,
        .element_count = scan_semantic.element_count,
        .block_size = scan_semantic.block_size,
        .count_source = scan_semantic.count_source,
    };
    const rund::kernel::ScanHash hash = rund::kernel::HashScan(descriptor);
    rund::kernel::ArtifactKey key = source.key;
    key.variant = rund::kernel::LoweringArtifactVariant::DeviceVsm;
    key.op_hash_hi =
        source.key.op_hash_hi ^ std::rotl(hash.hi, 11) ^ 0x677261706873636eull;
    key.op_hash_lo =
        source.key.op_hash_lo ^ std::rotl(hash.lo, 29) ^ 0x6d61702d7363616eull;
    key.canonical_ir_hash_hi = source.key.canonical_ir_hash_hi ^ hash.hi;
    key.canonical_ir_hash_lo = source.key.canonical_ir_hash_lo ^ hash.lo;
    std::string text = source.key.api == rund::kernel::ComputeApi::Metal
                           ? device_vsm_graph_map_scan::metal_source(
                                 key, input.parsed, scan_semantic.op)
                           : device_vsm_graph_map_scan::vulkan_source(
                                 key, input.parsed, scan_semantic.op);
    if (text.empty()) {
      result.reason = "device_vsm_graph_scan_source_invalid";
      return result;
    }
    result.artifact = source;
    result.artifact.key = key;
    result.artifact.kind =
        source.key.api == rund::kernel::ComputeApi::Metal
            ? rund::kernel::LoweringArtifactKind::MetalSource
            : rund::kernel::LoweringArtifactKind::VulkanSource;
    result.artifact.source_text = std::move(text);
    result.artifact.source_text_upper_bytes =
        result.artifact.source_text.size();
    result.artifact.ok = true;
    result.artifact.reason = "ok";
    result.plan = rund::kernel::ComputePlan{
        .tile_count = shape.elements,
        .op_hash_hi = key.op_hash_hi,
        .op_hash_lo = key.op_hash_lo,
        .api = source.key.api,
        .scalar = rund::kernel::ComputeScalar::Lane64,
        .domain = rund::kernel::ComputeDomain::U64,
        .input_buffer_count = input_count,
        .output_buffer_count = 1u,
        .input_bytes_per_tile = input_bytes_per_tile,
        .output_bytes_per_tile = shape.element_bytes,
        .param_bytes = source.metadata.param_storage.size(),
        .bytes_per_tile = bytes_per_tile,
        .dispatch_window_tiles = 1u,
        .dispatch_count = 1u,
        .fixed_authoritative = true,
        .ok = true,
        .reason = "ok",
    };
    result.proof = DeviceVsmScanProof{
        .semantic = scan_semantic,
        .map =
            DeviceVsmScanMap{.kind = DeviceVsmScanMapKind::CanonicalTotalU64},
        .workgroup_width = 256u,
        .stage_count = 2u,
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
