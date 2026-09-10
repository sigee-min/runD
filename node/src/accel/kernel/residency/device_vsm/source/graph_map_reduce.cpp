#include "graph_map_reduce.hpp"

#include "../graph_wavefront.hpp"
#include "graph_map_reduce/internal.hpp"

#include "../../../../../hash/fnv.hpp"

#include <bit>
#include <limits>
#include <new>

namespace rund::node::accel::detail {

DeviceVsmGraphMapReduceArtifact BuildDeviceVsmGraphMapReduceArtifact(
    const rund::kernel::LoweringArtifact &source,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &input,
    const MapSemantic &map_semantic,
    const rund::kernel::ReducePlan &reduce_semantic,
    const DeviceVsmPageGeometry &geometry,
    const DeviceVsmGraphWavefrontProof &wavefront) noexcept {
  DeviceVsmGraphMapReduceArtifact result{};
  if (!source.ok) {
    result.reason = "device_vsm_graph_source_unavailable";
    return result;
  }
  const std::uint64_t input_count = source.metadata.read_count;
  if (input_count == 0u || input_count >= DeviceVsmResidentCapacity ||
      (geometry.element_bytes != 0u &&
       input_count > std::numeric_limits<std::uint64_t>::max() /
                         geometry.element_bytes)) {
    result.reason = "device_vsm_graph_input_count_invalid";
    return result;
  }
  if (!device_vsm_graph_wavefront_valid(wavefront, geometry.page_count)) {
    result.reason = "device_vsm_graph_wavefront_invalid";
    return result;
  }
  if (!device_vsm_complete_frame_geometry(geometry)) {
    result.reason = "device_vsm_graph_geometry_invalid";
    return result;
  }
  if (geometry.logical_bytes / geometry.element_bytes >
      std::numeric_limits<std::uint32_t>::max()) {
    result.reason = "device_vsm_graph_element_count_invalid";
    return result;
  }
  try {
    if (!device_vsm_graph_map_reduce::validate(source, input, map_semantic,
                                               reduce_semantic)) {
      result.reason = "device_vsm_graph_ir_invalid";
      return result;
    }
    ::rund::node::hash_detail::Fnv wavefront_hi{
        ::rund::node::hash_detail::kFnvStandardOffset};
    ::rund::node::hash_detail::Fnv wavefront_lo{};
    const auto mix = [&](const std::uint64_t value) noexcept {
      wavefront_hi.Number(value);
      wavefront_lo.Number(value ^ 0x9e3779b97f4a7c15ull);
    };
    mix(wavefront.stage_count);
    mix(wavefront.frame_capacity);
    mix(wavefront.batch_count);
    mix(wavefront.map_stage);
    mix(wavefront.collective_stage);
    for (std::size_t stage = 0u; stage < DeviceVsmGraphStageCapacity; ++stage) {
      mix(wavefront.same_dispatch[stage]);
      mix(wavefront.same_release[stage]);
      mix(wavefront.prior_dispatch[stage]);
      mix(wavefront.prior_release[stage]);
    }
    const std::uint64_t wavefront_hash_hi = wavefront_hi.Finish();
    const std::uint64_t wavefront_hash_lo = wavefront_lo.Finish();
    rund::kernel::ArtifactKey key = source.key;
    key.variant = rund::kernel::LoweringArtifactVariant::DeviceVsm;
    key.op_hash_hi = source.key.op_hash_hi ^ std::rotl(wavefront_hash_hi, 13) ^
                     0x67726170682d7766ull;
    key.op_hash_lo = source.key.op_hash_lo ^ std::rotl(wavefront_hash_lo, 31) ^
                     0x6465766963657673ull;
    key.canonical_ir_hash_hi =
        source.key.canonical_ir_hash_hi ^ wavefront_hash_hi;
    key.canonical_ir_hash_lo =
        source.key.canonical_ir_hash_lo ^ wavefront_hash_lo;
    std::string text;
    if (source.key.api == rund::kernel::ComputeApi::Metal) {
      text = device_vsm_graph_map_reduce::metal_source(
          key, input.parsed, reduce_semantic.op, wavefront);
    } else if (source.key.api == rund::kernel::ComputeApi::Vulkan) {
      text = device_vsm_graph_map_reduce::vulkan_source(
          key, input.parsed, reduce_semantic.op, wavefront);
    }
    if (text.empty()) {
      result.reason = "device_vsm_graph_backend_invalid";
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
        .tile_count = geometry.logical_bytes / geometry.element_bytes,
        .op_hash_hi = key.op_hash_hi,
        .op_hash_lo = key.op_hash_lo,
        .api = source.key.api,
        .scalar = rund::kernel::ComputeScalar::Lane64,
        .domain = rund::kernel::ComputeDomain::U64,
        .input_buffer_count = input_count,
        .output_buffer_count = 1u,
        .input_bytes_per_tile = input_count * geometry.element_bytes,
        .output_bytes_per_tile = geometry.element_bytes,
        .param_bytes = source.metadata.param_storage.size(),
        .bytes_per_tile = (input_count + 1u) * geometry.element_bytes,
        .dispatch_window_tiles =
            geometry.logical_bytes / geometry.element_bytes,
        .dispatch_count = 1u,
        .ok = true,
        .reason = "ok",
    };
    result.proof = DeviceVsmGraphMapReduceProof{
        .semantic = reduce_semantic,
        .wavefront = wavefront,
        .workgroup_width = 256u,
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
