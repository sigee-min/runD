#include "proof.hpp"

#include <limits>

namespace rund::node::accel::detail {

bool device_vsm_window_ring_fusion_valid(
    const DeviceVsmWindowFusion &fusion) noexcept {
  const auto empty = [](const DeviceVsmWindowMap &map) noexcept {
    return map.kind == DeviceVsmWindowMapKind::None && map.immediate == 0u &&
           map.source_hi == 0u && map.source_lo == 0u &&
           map.canonical_hi == 0u && map.canonical_lo == 0u;
  };
  const auto immediate = [](const DeviceVsmWindowMap &map,
                            const DeviceVsmWindowMapKind kind,
                            const std::uint32_t value) noexcept {
    return map.kind == kind && map.immediate == value && map.source_hi == 0u &&
           map.source_lo == 0u && map.canonical_hi == 0u &&
           map.canonical_lo == 0u;
  };
  const auto canonical = [](const DeviceVsmWindowMap &map) noexcept {
    return map.kind == DeviceVsmWindowMapKind::CanonicalTotalU32 &&
           map.immediate == 0u && (map.source_hi != 0u || map.source_lo != 0u) &&
           (map.canonical_hi != 0u || map.canonical_lo != 0u);
  };
  const auto chain = [&](const DeviceVsmWindowMap &first,
                         const DeviceVsmWindowMap &second,
                         const DeviceVsmWindowMap &third,
                         const std::uint32_t depth) noexcept {
    return depth == 1u
               ? canonical(first) && empty(second) && empty(third)
           : depth == 2u
               ? canonical(first) && canonical(second) && empty(third)
               : depth == 3u && canonical(first) && canonical(second) &&
                     canonical(third);
  };
  const bool bare = empty(fusion.before) && empty(fusion.before_second) &&
                    empty(fusion.before_third) && empty(fusion.after) &&
                    empty(fusion.after_second) && empty(fusion.after_third) &&
                    fusion.stage_count == 1u;
  const bool immediate_fused =
      immediate(fusion.before, DeviceVsmWindowMapKind::AddWrapU32Immediate,
                3u) &&
      empty(fusion.before_second) && empty(fusion.before_third) &&
      immediate(fusion.after, DeviceVsmWindowMapKind::MulWrapU32Immediate,
                2u) &&
      empty(fusion.after_second) && empty(fusion.after_third) &&
      fusion.stage_count == 3u;
  const bool canonical_fused =
      (chain(fusion.before, fusion.before_second, fusion.before_third, 1u) &&
       chain(fusion.after, fusion.after_second, fusion.after_third, 1u) &&
       fusion.stage_count == 3u) ||
      (chain(fusion.before, fusion.before_second, fusion.before_third, 2u) &&
       chain(fusion.after, fusion.after_second, fusion.after_third, 2u) &&
       fusion.stage_count == 5u) ||
      (chain(fusion.before, fusion.before_second, fusion.before_third, 3u) &&
       chain(fusion.after, fusion.after_second, fusion.after_third, 3u) &&
       fusion.stage_count == 7u);
  return bare || immediate_fused || canonical_fused;
}

bool device_vsm_window_proof_valid(const DeviceVsmProof &proof) noexcept {
  const DeviceVsmWindowProof &window = proof.window;
  const rund::kernel::WindowPlan &semantic = window.semantic;
  const std::uint64_t element = proof.geometry.element_bytes;
  const std::uint64_t frame_elements = proof.geometry.frame_bytes / element;
  const std::uint64_t payload_elements = proof.geometry.payload_bytes / element;
  const std::uint64_t radius = proof.geometry.read_prefix_bytes / element;
  const bool width = window.workgroup_width == 64u ||
                     window.workgroup_width == 128u ||
                     window.workgroup_width == 256u;
  const DeviceVsmWindowFusion &fusion = window.fusion;
  const bool direct = window.range_path == RangePath::Direct;
  const bool shared = window.range_path == RangePath::SharedHalo;
  const bool prefix = window.range_path == RangePath::PrefixDifference;
  const bool block = window.range_path == RangePath::BlockPrefixSuffix;
  const bool ring_i32 =
      window.ring.gpu_owned &&
      semantic.element == rund::kernel::WindowElement::U32 &&
      semantic.domain == rund::kernel::ComputeDomain::I32;
  const bool scalar =
      window.ring.gpu_owned
          ? semantic.element == rund::kernel::WindowElement::U32 &&
                (semantic.domain == rund::kernel::ComputeDomain::U32 ||
                 semantic.domain == rund::kernel::ComputeDomain::I32)
          : semantic.element == rund::kernel::WindowElement::U32 &&
                semantic.domain == rund::kernel::ComputeDomain::U32;
  DeviceVsmWindowRingPlan ring_plan{};
  DeviceVsmWindowRingResult ring_result{};
  const bool window_ring = window.ring.gpu_owned;
  const bool ring_valid =
      !window_ring ||
      (device_vsm_window_ring_plan_expected(proof.geometry, ring_plan) &&
       window.ring == ring_plan &&
       device_vsm_window_ring_result_expected(proof.geometry, ring_plan,
                                              ring_result) &&
       (direct || shared) && !window.mutates_input &&
       (ring_i32 ? !fusion.active()
                 : device_vsm_window_ring_fusion_valid(fusion)) &&
       semantic.boundary == rund::kernel::WindowBoundary::Clamp);
  const bool ring_induction =
      !window_ring ||
      (width &&
       frame_elements <=
           std::numeric_limits<std::uint32_t>::max() -
               (static_cast<std::uint64_t>(window.workgroup_width) - 1u) &&
       payload_elements <=
           std::numeric_limits<std::uint32_t>::max() -
               (static_cast<std::uint64_t>(window.workgroup_width) - 1u));
  const bool range_valid =
      ((direct || shared) && window.range_stage_count == 1u &&
       !window.mutates_input &&
       semantic.boundary == rund::kernel::WindowBoundary::Clamp) ||
      ((prefix || block) && window.range_stage_count >= 2u &&
       window.mutates_input &&
       semantic.boundary == rund::kernel::WindowBoundary::Clip &&
       (prefix ? semantic.op == rund::kernel::WindowOp::Sum
               : semantic.op == rund::kernel::WindowOp::Min ||
                     semantic.op == rund::kernel::WindowOp::Max));
  const auto valid_map = [](const DeviceVsmWindowMap &map) noexcept {
    return map.kind == DeviceVsmWindowMapKind::None
               ? map.immediate == 0u && map.source_hi == 0u &&
                     map.source_lo == 0u && map.canonical_hi == 0u &&
                     map.canonical_lo == 0u
           : map.kind == DeviceVsmWindowMapKind::CanonicalTotalU32
               ? map.immediate == 0u
               : (map.kind == DeviceVsmWindowMapKind::AddWrapU32Immediate ||
                  map.kind == DeviceVsmWindowMapKind::MulWrapU32Immediate) &&
                     map.source_hi == 0u && map.source_lo == 0u &&
                     map.canonical_hi == 0u && map.canonical_lo == 0u;
  };
  const bool fusion_valid =
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
       (semantic.element == rund::kernel::WindowElement::U32 &&
        semantic.domain == rund::kernel::ComputeDomain::U32));
  DeviceVsmWindowFootprintAuthority footprint{};
  const bool footprint_valid =
      device_vsm_project_window_footprint(proof.geometry, footprint) &&
      window.footprint == footprint;
  return proof.topology == DeviceVsmTopology::Window && semantic.ok && width &&
         fusion_valid && range_valid && footprint_valid && ring_valid &&
         ring_induction && device_vsm_centered_window_geometry(proof.geometry) &&
         proof.parameter_bytes == DeviceVsmWindowParameterBytes &&
         proof.plan.param_bytes == DeviceVsmWindowParameterBytes &&
         semantic.element_bytes == element &&
         semantic.input_count == frame_elements &&
         semantic.output_count == frame_elements &&
         semantic.input_bytes == proof.geometry.frame_bytes &&
         semantic.output_bytes == proof.geometry.frame_bytes &&
         semantic.stride == 1u && semantic.pad_left == radius &&
         semantic.window_size == 2u * radius + 1u && scalar &&
         semantic.count_source == rund::kernel::ComputeCountSource::Descriptor &&
         (window.range_source_hi != 0u || window.range_source_lo != 0u) &&
         (window.range_execution_hi != 0u || window.range_execution_lo != 0u) &&
         window.shared_halo == shared &&
         (!window.shared_halo ||
          (semantic.boundary == rund::kernel::WindowBoundary::Clamp &&
           window.shared_radius_capacity >= radius)) &&
         (window.shared_halo || window.shared_radius_capacity == 0u);
}

} // namespace rund::node::accel::detail
