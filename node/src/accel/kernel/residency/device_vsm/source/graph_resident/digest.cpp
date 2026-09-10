#include "internal.hpp"

#include "../../../../../../hash/fnv.hpp"

namespace rund::node::accel::detail::device_vsm_graph_resident_source::detail {

bool graph_resident_key(
    const std::span<const DeviceVsmGraphResidentStageSource> stages,
    const DeviceVsmGraphResidentProof &proof,
    const DeviceVsmGraphWavefrontProof &wavefront,
    rund::kernel::ArtifactKey &key) noexcept {
  if (stages.empty() || stages.front().artifact == nullptr) {
    return false;
  }
  ::rund::node::hash_detail::Fnv hi{
      ::rund::node::hash_detail::kFnvStandardOffset};
  ::rund::node::hash_detail::Fnv lo{};
  const auto mix = [&](const std::uint64_t value) noexcept {
    hi.Number(value);
    lo.Number(value ^ 0x9e3779b97f4a7c15ull);
  };
  if (!device_vsm_graph_resident_type_valid(proof.type)) {
    return false;
  }
  mix(static_cast<std::uint64_t>(proof.type.scalar));
  mix(static_cast<std::uint64_t>(proof.type.domain));
  mix(proof.type.element_bytes);
  for (const auto &stage : stages) {
    mix(stage.artifact->key.op_hash_hi);
    mix(stage.artifact->key.op_hash_lo);
    mix(stage.artifact->key.canonical_ir_hash_hi);
    mix(stage.artifact->key.canonical_ir_hash_lo);
  }
  const auto &first = *stages.front().artifact;
  mix(proof.digest);
  for (const std::uint32_t word : proof.page_map.words) {
    mix(word);
  }
  mix(proof.owner_binding_count);
  mix(wavefront.stage_count);
  mix(wavefront.frame_capacity);
  mix(wavefront.batch_count);
  mix(wavefront.map_stage);
  mix(wavefront.collective_stage);
  mix(DeviceVsmGraphControllerVersion);
  mix(0x475241504854494Cull);
  for (std::size_t stage = 0u; stage < wavefront.stage_count; ++stage) {
    mix(wavefront.same_dispatch[stage]);
    mix(wavefront.same_release[stage]);
    mix(wavefront.prior_dispatch[stage]);
    mix(wavefront.prior_release[stage]);
  }
  key = first.key;
  key.scalar = proof.type.scalar;
  key.domain = proof.type.domain;
  key.variant = rund::kernel::LoweringArtifactVariant::DeviceVsm;
  key.op_hash_hi = hi.Finish();
  key.op_hash_lo = lo.Finish();
  key.canonical_ir_hash_hi = hi.Finish();
  key.canonical_ir_hash_lo = lo.Finish();
  return true;
}

} // namespace rund::node::accel::detail::device_vsm_graph_resident_source::detail
