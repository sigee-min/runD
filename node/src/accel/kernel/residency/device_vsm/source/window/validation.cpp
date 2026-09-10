#include "internal.hpp"

#include "../../validation.hpp"

#include "../typed_map/scalar.hpp"

#include "../../../../../range_aggregate/execution/projection.hpp"
#include "../../../../../window/shape.hpp"

namespace rund::node::accel::detail {
namespace device_vsm_window_source {
namespace {

[[nodiscard]] bool valid_map(const DeviceVsmWindowMap &map) noexcept {
  if (map.kind == DeviceVsmWindowMapKind::None) {
    return map.immediate == 0u && map.source_hi == 0u && map.source_lo == 0u &&
           map.canonical_hi == 0u && map.canonical_lo == 0u;
  }
  if (map.kind == DeviceVsmWindowMapKind::CanonicalTotalU32) {
    return map.immediate == 0u;
  }
  return (map.kind == DeviceVsmWindowMapKind::AddWrapU32Immediate ||
          map.kind == DeviceVsmWindowMapKind::MulWrapU32Immediate) &&
         map.source_hi == 0u && map.source_lo == 0u && map.canonical_hi == 0u &&
         map.canonical_lo == 0u;
}

[[nodiscard]] bool
valid_map_source(const DeviceVsmWindowMap &map,
                 const DeviceVsmWindowMapSource &source) noexcept {
  if (map.kind != DeviceVsmWindowMapKind::CanonicalTotalU32) {
    return source.artifact == nullptr && source.input == nullptr;
  }
  return source.artifact != nullptr && source.input != nullptr &&
         map.source_hi == source.artifact->key.op_hash_hi &&
         map.source_lo == source.artifact->key.op_hash_lo &&
         map.canonical_hi == source.artifact->key.canonical_ir_hash_hi &&
         map.canonical_lo == source.artifact->key.canonical_ir_hash_lo &&
         device_vsm_typed_map::validate_parameter_free_total_u32_scalar(
             *source.artifact, *source.input);
}

[[nodiscard]] bool exact_geometry(
    const RangeExec &execution, const rund::kernel::WindowPlan &semantic,
    const DeviceVsmPageGeometry &geometry, const DeviceVsmWindowFusion &fusion,
    const DeviceVsmWindowRingPlan &ring,
    const DeviceVsmWindowMapSources &sources) noexcept {
  const std::uint64_t element = geometry.element_bytes;
  const std::uint64_t frame_elements = geometry.frame_bytes / element;
  const std::uint64_t radius = geometry.read_prefix_bytes / element;
  const bool ring_i32 = ring.gpu_owned && !execution.wide_elements() &&
                        semantic.element == rund::kernel::WindowElement::U32 &&
                        semantic.domain == rund::kernel::ComputeDomain::I32;
  const bool ring_u32 = ring.gpu_owned && !execution.wide_elements() &&
                        semantic.element == rund::kernel::WindowElement::U32 &&
                        semantic.domain == rund::kernel::ComputeDomain::U32;
  const bool scalar_ok =
      ring.gpu_owned ? ring_i32 || ring_u32
                     : semantic.element == rund::kernel::WindowElement::U32 &&
                           semantic.domain == rund::kernel::ComputeDomain::U32;
  const bool source_valid =
      valid_map_source(fusion.before, sources.before) &&
      valid_map_source(fusion.before_second, sources.before_second) &&
      valid_map_source(fusion.before_third, sources.before_third) &&
      valid_map_source(fusion.after, sources.after) &&
      valid_map_source(fusion.after_second, sources.after_second) &&
      valid_map_source(fusion.after_third, sources.after_third);
  const bool generic_fusion_shape =
      valid_map(fusion.before) && valid_map(fusion.before_second) &&
      valid_map(fusion.before_third) && valid_map(fusion.after) &&
      valid_map(fusion.after_second) && valid_map(fusion.after_third) &&
      (!fusion.before_second.active() || fusion.before.active()) &&
      (!fusion.before_third.active() || fusion.before_second.active()) &&
      (!fusion.after_second.active() || fusion.after.active()) &&
      (!fusion.after_third.active() || fusion.after_second.active()) &&
      fusion.stage_count ==
          1u + static_cast<std::uint32_t>(fusion.before.active()) +
              static_cast<std::uint32_t>(fusion.before_second.active()) +
              static_cast<std::uint32_t>(fusion.before_third.active()) +
              static_cast<std::uint32_t>(fusion.after.active()) +
              static_cast<std::uint32_t>(fusion.after_second.active()) +
              static_cast<std::uint32_t>(fusion.after_third.active()) &&
      (!fusion.active() ||
       (!execution.wide_elements() && !execution.signed_values()));
  const bool ring_fusion_shape = device_vsm_window_ring_fusion_valid(fusion) &&
                                 (!ring_i32 || !fusion.active());
  const bool fusion_shape =
      ring.gpu_owned ? ring_fusion_shape : generic_fusion_shape;
  DeviceVsmWindowRingPlan expected_ring{};
  const bool ring_shape =
      !ring.gpu_owned ||
      (device_vsm_window_ring_plan_expected(geometry, expected_ring) &&
       ring == expected_ring &&
       (execution.candidate() == RangePath::Direct ||
        execution.candidate() == RangePath::SharedHalo) &&
       !DeviceVsmWindowRangeMutatesInput(execution) &&
       semantic.boundary == rund::kernel::WindowBoundary::Clamp);
  return source_valid && fusion_shape && ring_shape &&
         device_vsm_runtime_geometry_valid(geometry) &&
         device_vsm_centered_window_geometry(geometry) && semantic.ok &&
         semantic.element_bytes == element &&
         semantic.input_count == frame_elements &&
         semantic.output_count == frame_elements && semantic.stride == 1u &&
         semantic.pad_left == radius &&
         semantic.window_size == 2u * radius + 1u &&
         WindowRangePlanMatches(semantic, execution.plan()) && scalar_ok &&
         !execution.wide_elements() &&
         DeviceVsmWindowRangeSupported(execution) &&
         (!execution.uses_shared_halo() ||
          execution.shared_radius_capacity() >= radius);
}

} // namespace

bool validate_geometry(const RangeExec &execution,
                       const rund::kernel::WindowPlan &semantic,
                       const DeviceVsmPageGeometry &geometry,
                       const DeviceVsmWindowFusion &fusion,
                       const DeviceVsmWindowRingPlan &ring,
                       const DeviceVsmWindowMapSources &sources) noexcept {
  return exact_geometry(execution, semantic, geometry, fusion, ring, sources);
}

} // namespace device_vsm_window_source

bool DeviceVsmWindowRangeSupported(const RangeExec &execution) noexcept {
  const RangePath path = execution.candidate();
  const std::size_t stages = execution.plan().stage_count();
  if (path == RangePath::Direct || path == RangePath::SharedHalo) {
    return stages == 1u &&
           execution.plan().shape().boundary() == RangeBoundary::Clamp;
  }
  if (execution.plan().shape().boundary() != RangeBoundary::Clip ||
      stages < 2u) {
    return false;
  }
  return path == RangePath::PrefixDifference
             ? execution.operation() == RangeOp::Sum
             : path == RangePath::BlockPrefixSuffix &&
                   (execution.operation() == RangeOp::Minimum ||
                    execution.operation() == RangeOp::Maximum);
}

bool DeviceVsmWindowRangeMutatesInput(const RangeExec &execution) noexcept {
  return DeviceVsmWindowRangeSupported(execution) &&
         (execution.candidate() == RangePath::PrefixDifference ||
          execution.candidate() == RangePath::BlockPrefixSuffix);
}

} // namespace rund::node::accel::detail
