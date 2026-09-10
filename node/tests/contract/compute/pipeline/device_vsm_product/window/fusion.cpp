#include "fusion.hpp"

namespace rund_node_test_device_vsm_product::window_test {
namespace {

using MapAuthority = rund::node::accel::detail::DeviceVsmWindowMap;
using MapKind = rund::node::accel::detail::DeviceVsmWindowMapKind;

[[nodiscard]] bool canonical_map(const MapAuthority &map) noexcept {
  return map.kind == MapKind::CanonicalTotalU32 &&
         (map.source_hi != 0u || map.source_lo != 0u) &&
         (map.canonical_hi != 0u || map.canonical_lo != 0u);
}

[[nodiscard]] bool immediate_map(const MapAuthority &map, const MapKind kind,
                                 const std::uint64_t immediate) noexcept {
  return map.kind == kind && map.immediate == immediate;
}

} // namespace

bool ExactWindowFusion(const rund::node::accel::detail::DeviceVsmProof &proof,
                       const PipelineShape shape) noexcept {
  const auto &fusion = proof.window.fusion;
  if (shape == PipelineShape::Window) {
    return !fusion.active() && fusion.stage_count == 1u;
  }
  if (shape == PipelineShape::MapWindowMap) {
    return immediate_map(fusion.before, MapKind::AddWrapU32Immediate, 3u) &&
           immediate_map(fusion.after, MapKind::MulWrapU32Immediate, 2u) &&
           !fusion.before_second.active() && !fusion.after_second.active() &&
           !fusion.before_third.active() && !fusion.after_third.active() &&
           fusion.stage_count == 3u;
  }
  if (shape == PipelineShape::MapDagWindowMapDag) {
    return canonical_map(fusion.before) && canonical_map(fusion.after) &&
           !fusion.before_second.active() && !fusion.after_second.active() &&
           !fusion.before_third.active() && !fusion.after_third.active() &&
           fusion.stage_count == 3u;
  }
  const bool first_two =
      canonical_map(fusion.before) && canonical_map(fusion.before_second) &&
      canonical_map(fusion.after) && canonical_map(fusion.after_second);
  if (shape == PipelineShape::MapChainWindowMapChain) {
    return first_two && !fusion.before_third.active() &&
           !fusion.after_third.active() && fusion.stage_count == 5u;
  }
  return first_two && canonical_map(fusion.before_third) &&
         canonical_map(fusion.after_third) && fusion.stage_count == 7u;
}

} // namespace rund_node_test_device_vsm_product::window_test
