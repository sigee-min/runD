#include "reduce.hpp"

#include "reduce/internal.hpp"

#include <kernel/program/compute/reduce/identity.hpp>
#include <kernel/program/compute/reduce/plan.hpp>

#include <limits>
#include <new>

namespace rund::node::accel::detail {

DeviceVsmReduceArtifact
BuildDeviceVsmReduceArtifact(const rund::kernel::ReducePlan &semantic,
                             const rund::kernel::ComputeApi api,
                             const DeviceVsmPageGeometry &geometry) noexcept {
  DeviceVsmReduceArtifact result{};
  const bool backend = api == rund::kernel::ComputeApi::Metal ||
                       api == rund::kernel::ComputeApi::Vulkan;
  const bool u32 = semantic.element == rund::kernel::ReduceElement::U32;
  const bool u64 = semantic.element == rund::kernel::ReduceElement::U64;
  const bool operation = semantic.op == rund::kernel::ReduceOp::Sum ||
                         semantic.op == rund::kernel::ReduceOp::CountNonzero ||
                         semantic.op == rund::kernel::ReduceOp::Min ||
                         semantic.op == rund::kernel::ReduceOp::Max;
  const std::uint64_t element_bytes =
      u32 ? sizeof(std::uint32_t) : sizeof(std::uint64_t);
  const std::uint64_t elements =
      geometry.element_bytes == 0u
          ? 0u
          : geometry.logical_bytes / geometry.element_bytes;
  if (!semantic.ok || !backend || !operation || (!u32 && !u64) ||
      semantic.count_source != rund::kernel::ComputeCountSource::Descriptor ||
      semantic.element_bytes != element_bytes ||
      semantic.element_bytes != geometry.element_bytes ||
      semantic.element_count != geometry.frame_bytes / element_bytes ||
      !device_vsm_runtime_geometry_valid(geometry) ||
      !device_vsm_complete_frame_geometry(geometry) || elements == 0u ||
      elements > std::numeric_limits<std::uint32_t>::max()) {
    result.reason = "device_vsm_reduce_geometry_invalid";
    return result;
  }
  try {
    const rund::kernel::ReduceDesc descriptor{
        .op = semantic.op,
        .element = semantic.element,
        .element_count = semantic.element_count,
        .block_size = semantic.block_size,
        .count_source = semantic.count_source,
    };
    if (!rund::kernel::ReducePlanMatchesDesc(descriptor, semantic)) {
      result.reason = "device_vsm_reduce_semantic_invalid";
      return result;
    }
    const rund::kernel::ReduceHash hash = rund::kernel::HashReduce(descriptor);
    const rund::kernel::ArtifactKey key{
        .api = api,
        .scalar = u32 ? rund::kernel::ComputeScalar::Lane32
                      : rund::kernel::ComputeScalar::Lane64,
        .domain = u32 ? rund::kernel::ComputeDomain::U32
                      : rund::kernel::ComputeDomain::U64,
        .variant = rund::kernel::LoweringArtifactVariant::DeviceVsm,
        .op_hash_hi = hash.hi ^ 0x6465766963657673ull,
        .op_hash_lo = hash.lo ^ 0x6d2e726564756365ull,
        .canonical_ir_hash_hi = hash.hi,
        .canonical_ir_hash_lo = hash.lo,
    };
    std::string source = api == rund::kernel::ComputeApi::Metal
                             ? device_vsm_reduce_source::metal_source(
                                   key, semantic.element, semantic.op)
                             : device_vsm_reduce_source::vulkan_source(
                                   key, semantic.element, semantic.op);
    if (source.empty()) {
      result.reason = "device_vsm_reduce_source_invalid";
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
        .tile_count = elements,
        .op_hash_hi = key.op_hash_hi,
        .op_hash_lo = key.op_hash_lo,
        .api = api,
        .scalar = key.scalar,
        .domain = key.domain,
        .input_buffer_count = 1u,
        .output_buffer_count = 1u,
        .input_bytes_per_tile = element_bytes,
        .output_bytes_per_tile = element_bytes,
        .bytes_per_tile = element_bytes * 2u,
        .dispatch_window_tiles = 1u,
        .dispatch_count = 1u,
        .fixed_authoritative = true,
        .ok = true,
        .reason = "ok",
    };
    result.proof =
        DeviceVsmReduceProof{.semantic = semantic, .workgroup_width = 256u};
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
