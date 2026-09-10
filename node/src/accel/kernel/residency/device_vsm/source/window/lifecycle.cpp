#include "internal.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace rund::node::accel::detail::device_vsm_window_source {

bool materialize_artifact(
    DeviceVsmWindowArtifact &result, const RangeExec &execution,
    const rund::kernel::WindowPlan &semantic,
    const DeviceVsmPageGeometry &geometry, const DeviceVsmWindowFusion &fusion,
    const DeviceVsmWindowRingPlan &ring, const rund::kernel::ArtifactKey &key,
    const rund::kernel::ComputeApi api, std::string &&source_text,
    const std::uint64_t payload_elements, const std::uint64_t logical_elements,
    const std::uint64_t bytes_per_tile, const std::uint64_t staging_bytes,
    const bool multipass) noexcept {
  try {
    result.artifact.key = key;
    result.artifact.kind =
        api == rund::kernel::ComputeApi::Metal
            ? rund::kernel::LoweringArtifactKind::MetalSource
            : rund::kernel::LoweringArtifactKind::VulkanSource;
    result.artifact.metadata.ok = true;
    result.artifact.metadata.read_count = 1u;
    result.artifact.metadata.write_count = 1u;
    result.artifact.source_text = std::move(source_text);
    result.artifact.source_text_upper_bytes =
        result.artifact.source_text.size();
    result.artifact.ok = true;
    result.artifact.reason = "ok";
    result.plan = rund::kernel::ComputePlan{
        .tile_count = payload_elements,
        .op_hash_hi = key.op_hash_hi,
        .op_hash_lo = key.op_hash_lo,
        .api = api,
        .scalar = key.scalar,
        .domain = key.domain,
        .input_buffer_count = 1u,
        .output_buffer_count = 1u,
        .input_bytes_per_tile = geometry.element_bytes,
        .output_bytes_per_tile = geometry.element_bytes,
        .param_bytes = DeviceVsmWindowParameterBytes,
        .bytes_per_tile = bytes_per_tile,
        .staging_bytes = staging_bytes,
        .dispatch_window_tiles = payload_elements,
        .dispatch_count = 1u,
        .fixed_authoritative = true,
        .ok = true,
        .reason = "ok",
    };
    const Parameters parameters{
        .input_count = logical_elements,
        .output_count = logical_elements,
        .window_size = semantic.window_size,
        .stride = semantic.stride,
        .padding = semantic.pad_left,
        .stage_element_count = logical_elements,
        .stage_aux_count =
            (logical_elements + execution.width() - 1u) / execution.width(),
        .stage = static_cast<std::uint32_t>(execution.uses_shared_halo()
                                                ? RangeStageKind::SharedHalo
                                                : RangeStageKind::Direct),
    };
    std::memcpy(result.parameters.data(), &parameters, sizeof(parameters));
    result.window = rund::kernel::ComputeDispatchWindow{
        .begin_sequence = 0u,
        .tile_count = payload_elements,
    };
    result.proof = DeviceVsmWindowProof{
        .semantic = semantic,
        .fusion = fusion,
        .ring = ring,
        .footprint =
            [&geometry]() noexcept {
              DeviceVsmWindowFootprintAuthority authority{};
              static_cast<void>(
                  device_vsm_project_window_footprint(geometry, authority));
              return authority;
            }(),
        .range_source_hi = execution.source_identity().hi,
        .range_source_lo = execution.source_identity().lo,
        .range_execution_hi = execution.execution_identity().hi,
        .range_execution_lo = execution.execution_identity().lo,
        .workgroup_width = execution.width(),
        .shared_radius_capacity = execution.uses_shared_halo()
                                      ? execution.shared_radius_capacity()
                                      : 0u,
        .range_path = execution.candidate(),
        .range_stage_count = static_cast<std::uint8_t>(
            std::min<std::size_t>(execution.plan().stage_count(),
                                  std::numeric_limits<std::uint8_t>::max())),
        .shared_halo = execution.uses_shared_halo(),
        .mutates_input = multipass,
    };
    result.ok = true;
    result.reason = "ok";
    return true;
  } catch (const std::bad_alloc &) {
    result.reason = "compute_pipeline_capacity";
    return false;
  } catch (const std::length_error &) {
    result.reason = "compute_pipeline_capacity";
    return false;
  }
}

} // namespace rund::node::accel::detail::device_vsm_window_source
