#include "window/internal.hpp"

#include "../../../../range_aggregate/execution/projection.hpp"

#include <kernel/core/checked.hpp>

#include <limits>
#include <new>
#include <optional>
#include <stdexcept>
#include <utility>

namespace rund::node::accel::detail {

DeviceVsmWindowArtifact BuildDeviceVsmWindowArtifact(
    const RangePlan &range, const rund::kernel::WindowPlan &semantic,
    const DeviceVsmPageGeometry &geometry, const DeviceVsmWindowFusion fusion,
    const DeviceVsmWindowRingPlan ring,
    const DeviceVsmWindowMapSources sources) noexcept {
  using namespace device_vsm_window_source;
  DeviceVsmWindowArtifact result{};
  const std::optional<RangeExec> execution = RangeExec::from(range);
  if (!execution.has_value() ||
      !validate_geometry(*execution, semantic, geometry, fusion, ring,
                         sources)) {
    result.reason = "device_vsm_window_geometry_invalid";
    return result;
  }
  const RangeIdentity source = execution->source_identity();
  const RangeIdentity physical = execution->execution_identity();
  const rund::kernel::ComputeApi api = api_for(range.source_variant());
  const std::uint64_t payload_elements =
      geometry.payload_bytes / geometry.element_bytes;
  const std::uint64_t logical_elements =
      geometry.logical_bytes / geometry.element_bytes;
  std::uint64_t bytes_per_tile = 0u;
  std::uint64_t staging_bytes = 0u;
  if (payload_elements == 0u || logical_elements == 0u ||
      logical_elements > std::numeric_limits<std::uint32_t>::max() ||
      !::rund::kernel::checked::add(geometry.element_bytes,
                                    geometry.element_bytes, bytes_per_tile) ||
      !::rund::kernel::checked::add(geometry.payload_bytes,
                                    geometry.payload_bytes, staging_bytes) ||
      !::rund::kernel::checked::add(
          staging_bytes, DeviceVsmWindowParameterBytes, staging_bytes)) {
    result.reason = "device_vsm_window_capacity";
    return result;
  }
  try {
    const rund::kernel::ComputeScalar scalar =
        execution->wide_elements() ? rund::kernel::ComputeScalar::Lane64
                                   : rund::kernel::ComputeScalar::Lane32;
    const rund::kernel::ComputeDomain domain = executable_domain(*execution);
    const rund::kernel::ArtifactKey key =
        artifact_key(source, physical, api, scalar, domain, fusion, ring);
    std::string source_text{};
    const bool multipass = DeviceVsmWindowRangeMutatesInput(*execution);
    const bool emitted =
        api == rund::kernel::ComputeApi::Metal
            ? (multipass ? emit_metal_multipass(*execution, key, fusion,
                                                sources, source_text)
                         : emit_metal(*execution, key, fusion, ring, sources,
                                      source_text))
            : (multipass ? emit_vulkan_multipass(*execution, key, fusion,
                                                 sources, source_text)
                         : emit_vulkan(*execution, key, fusion, ring, sources,
                                       source_text));
    if (!emitted || source_text.empty()) {
      result.reason = "device_vsm_window_source_invalid";
      return result;
    }
    if (!materialize_artifact(result, *execution, semantic, geometry, fusion,
                              ring, key, api, std::move(source_text),
                              payload_elements, logical_elements,
                              bytes_per_tile, staging_bytes, multipass)) {
      return result;
    }
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
