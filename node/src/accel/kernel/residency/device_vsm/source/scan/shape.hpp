#pragma once

#include "../../geometry.hpp"

#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/scan/plan.hpp>

#include <cstdint>
#include <limits>

namespace rund::node::accel::detail::device_vsm_scan_source {

struct Shape final {
  std::uint64_t element_bytes{};
  std::uint64_t elements{};
  bool inclusive{};
  bool u32{};
  bool u64{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return element_bytes != 0u && elements != 0u && (u32 || u64);
  }
};

[[nodiscard]] inline Shape
project_shape(const rund::kernel::ScanPlan &semantic,
              const rund::kernel::ComputeApi api,
              const DeviceVsmPageGeometry &geometry) noexcept {
  const bool backend = api == rund::kernel::ComputeApi::Metal ||
                       api == rund::kernel::ComputeApi::Vulkan;
  const bool inclusive = semantic.op == rund::kernel::ScanOp::InclusiveSum;
  const bool exclusive = semantic.op == rund::kernel::ScanOp::ExclusiveSum;
  const bool u32 = semantic.element == rund::kernel::ScanElement::U32;
  const bool u64 = semantic.element == rund::kernel::ScanElement::U64;
  const std::uint64_t element_bytes =
      u32 ? sizeof(std::uint32_t) : sizeof(std::uint64_t);
  const std::uint64_t elements =
      geometry.element_bytes == 0u
          ? 0u
          : geometry.logical_bytes / geometry.element_bytes;
  const bool exact_geometry =
      inclusive
          ? device_vsm_complete_frame_geometry(geometry)
          : geometry.read_prefix_bytes == element_bytes &&
                geometry.target_offset_bytes == element_bytes &&
                geometry.read_suffix_bytes == 0u &&
                geometry.frame_bytes == geometry.payload_bytes + element_bytes;
  const rund::kernel::ScanDesc descriptor{
      .op = semantic.op,
      .element = semantic.element,
      .element_count = semantic.element_count,
      .block_size = semantic.block_size,
      .count_source = semantic.count_source,
  };
  if (!semantic.ok || !backend || (!inclusive && !exclusive) ||
      (!u32 && !u64) ||
      semantic.count_source != rund::kernel::ComputeCountSource::Descriptor ||
      semantic.element_bytes != element_bytes ||
      semantic.element_bytes != geometry.element_bytes ||
      semantic.element_count != geometry.frame_bytes / element_bytes ||
      !device_vsm_runtime_geometry_valid(geometry) || elements == 0u ||
      elements > std::numeric_limits<std::uint32_t>::max() || !exact_geometry ||
      !rund::kernel::ScanPlanMatchesDesc(descriptor, semantic)) {
    return {};
  }
  return Shape{.element_bytes = element_bytes,
               .elements = elements,
               .inclusive = inclusive,
               .u32 = u32,
               .u64 = u64};
}

} // namespace rund::node::accel::detail::device_vsm_scan_source
