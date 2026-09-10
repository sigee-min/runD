#include "scan.hpp"

#include "scan/internal.hpp"
#include "scan/shape.hpp"

#include <kernel/program/compute/scan/identity.hpp>

#include <bit>
#include <new>

namespace rund::node::accel::detail {

DeviceVsmScanArtifact
BuildDeviceVsmScanArtifact(const rund::kernel::ScanPlan &semantic,
                           const rund::kernel::ComputeApi api,
                           const DeviceVsmPageGeometry &geometry,
                           const DeviceVsmScanMap map) noexcept {
  DeviceVsmScanArtifact result{};
  const device_vsm_scan_source::Shape shape =
      device_vsm_scan_source::project_shape(semantic, api, geometry);
  const bool valid_map =
      map.kind == DeviceVsmScanMapKind::None ||
      (shape.u64 && map.kind == DeviceVsmScanMapKind::AddWrapU64Immediate);
  if (!shape || !valid_map) {
    result.reason = "device_vsm_scan_geometry_invalid";
    return result;
  }
  try {
    const rund::kernel::ScanDesc descriptor{
        .op = semantic.op,
        .element = semantic.element,
        .element_count = semantic.element_count,
        .block_size = semantic.block_size,
        .count_source = semantic.count_source,
    };
    const rund::kernel::ScanHash hash = rund::kernel::HashScan(descriptor);
    const std::uint64_t map_hi =
        map.active() ? map.immediate ^ 0x6d61702d7363616eull : 0u;
    const std::uint64_t map_lo =
        map.active() ? std::rotl(map.immediate, 17) ^ 0x76312d6675736564ull
                     : 0u;
    const rund::kernel::ArtifactKey key{
        .api = api,
        .scalar = shape.u32 ? rund::kernel::ComputeScalar::Lane32
                            : rund::kernel::ComputeScalar::Lane64,
        .domain = shape.u32 ? rund::kernel::ComputeDomain::U32
                            : rund::kernel::ComputeDomain::U64,
        .variant = rund::kernel::LoweringArtifactVariant::DeviceVsm,
        .op_hash_hi = hash.hi ^ 0x6465766963657673ull ^ map_hi,
        .op_hash_lo = hash.lo ^ 0x6d2e7363616e2e31ull ^ map_lo,
        .canonical_ir_hash_hi = hash.hi ^ map_hi,
        .canonical_ir_hash_lo = hash.lo ^ map_lo,
    };
    std::string source =
        api == rund::kernel::ComputeApi::Metal
            ? (shape.u32 ? device_vsm_scan_source::metal_source_u32(
                               key, semantic.op, map)
                         : device_vsm_scan_source::metal_source_u64(
                               key, semantic.op, map))
            : (shape.u32 ? device_vsm_scan_source::vulkan_source_u32(
                               key, semantic.op, map)
                         : device_vsm_scan_source::vulkan_source_u64(
                               key, semantic.op, map));
    if (source.empty()) {
      result.reason = "device_vsm_scan_source_invalid";
      return result;
    }
    result.artifact.key = key;
    result.artifact.kind =
        api == rund::kernel::ComputeApi::Metal
            ? rund::kernel::LoweringArtifactKind::MetalSource
            : rund::kernel::LoweringArtifactKind::VulkanSource;
    result.artifact.metadata.ok = true;
    result.artifact.metadata.read_count = 1u;
    result.artifact.metadata.write_count = 1u;
    result.artifact.source_text = std::move(source);
    result.artifact.source_text_upper_bytes =
        result.artifact.source_text.size();
    result.artifact.ok = true;
    result.artifact.reason = "ok";
    result.plan = rund::kernel::ComputePlan{
        .tile_count = shape.elements,
        .op_hash_hi = key.op_hash_hi,
        .op_hash_lo = key.op_hash_lo,
        .api = api,
        .scalar = key.scalar,
        .domain = key.domain,
        .input_buffer_count = 1u,
        .output_buffer_count = 1u,
        .input_bytes_per_tile = shape.element_bytes,
        .output_bytes_per_tile = shape.element_bytes,
        .bytes_per_tile = shape.element_bytes * 2u,
        .dispatch_window_tiles = 1u,
        .dispatch_count = 1u,
        .fixed_authoritative = true,
        .ok = true,
        .reason = "ok",
    };
    result.proof = DeviceVsmScanProof{
        .semantic = semantic,
        .map = map,
        .workgroup_width = 256u,
        .stage_count = static_cast<std::uint32_t>(1u + map.active()),
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
