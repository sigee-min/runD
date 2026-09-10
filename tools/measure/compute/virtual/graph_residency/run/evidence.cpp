#include "local.hpp"

#include <span>

namespace rund::measure::compute::virtual_graph_residency::run_detail {

[[nodiscard]] bool alias_reuse(
    const ::rund::node::accel::detail::DeviceVsmGraphResidentProof &proof,
    const ::rund::node::accel::detail::DeviceVsmGraphWavefrontProof
        &wavefront) noexcept {
  using Resource = ::rund::node::accel::detail::DeviceVsmGraphResidentResource;
  const Resource *x = nullptr;
  const Resource *w = nullptr;
  for (std::size_t index = 0u; index < proof.resource_count; ++index) {
    const auto &resource = proof.resources[index];
    if (resource.first == 0u && resource.last == 2u) {
      x = &resource;
    }
    if (resource.first == 3u && resource.last == 4u) {
      w = &resource;
    }
  }
  return x != nullptr && w != nullptr && x->owner_slot == w->owner_slot &&
         x->physical_id == w->physical_id && x->color == w->color &&
         (wavefront.same_dispatch[3u] & (std::uint8_t{1u} << 2u)) != 0u &&
         (wavefront.same_release[3u] & (std::uint8_t{1u} << 2u)) == 0u;
}

[[nodiscard]] Facts
facts(const Case &test_case, const ::rund::compute::Status status,
      const virtual_residency::ProductRouteEvidence &route,
      const std::uint64_t version_before, const std::uint64_t version_after,
      const bool output_match, const std::vector<std::uint64_t> &values,
      const ::rund::compute::Stats &stats) {
  Facts result{};
  result.stats = stats;
  result.residency = stats.pipeline.residency;
  result.uploaded_bytes = stats.uploaded_bytes;
  result.downloaded_bytes = stats.downloaded_bytes;
  result.internal_roundtrip_bytes = stats.internal_roundtrip_bytes;
  result.external_roundtrip_bytes = stats.external_roundtrip_bytes;
  result.output_match = output_match;
  result.output_hash = Spec::hash(std::span{values});
  result.version_before = version_before;
  result.version_after = version_after;
  result.route = route;
  result.ok = static_cast<bool>(status) && output_match;
  const auto state = ::rund::compute::detail::VirtualPipelineAccess::state(
      *test_case.pipeline);
  if (state == nullptr) {
    return result;
  }
  const auto owner =
      std::static_pointer_cast<Owner>(state->device_vsm_product_cache);
  result.owner_present = owner != nullptr;
  if (owner == nullptr || owner->proof == nullptr ||
      owner->evidence == nullptr) {
    return result;
  }
  const auto &proof = *owner->proof;
  // Copy the terminal evidence before any later owner observation so each
  // packet row has one immutable native snapshot.
  const auto native = owner->evidence->native;
  result.final_received = owner->evidence->final_received;
  result.may_write = native.may_write;
  result.owner_identity = reinterpret_cast<std::uintptr_t>(owner.get());
  result.control_identity =
      reinterpret_cast<std::uintptr_t>(owner->registration.get());
  result.graph_resident =
      proof.topology ==
      ::rund::node::accel::detail::DeviceVsmTopology::GraphResident;
  result.proof_valid =
      result.graph_resident &&
      ::rund::node::accel::detail::device_vsm_graph_resident_valid(
          proof.graph_resident, Spec::FrameElements, Spec::InputCount) &&
      ::rund::node::accel::detail::device_vsm_graph_wavefront_valid(
          proof.graph_wavefront, Spec::PageCount);
  result.alias_reuse = result.graph_resident &&
                       alias_reuse(proof.graph_resident, proof.graph_wavefront);
  result.proof_digest = proof.graph_resident.digest;
  result.proof_hi = native.proof.hi;
  result.proof_lo = native.proof.lo;
  result.generation = native.generation;
  result.nonce = native.nonce;
  result.resource_count = proof.graph_resident.resource_count;
  result.owner_count = proof.graph_resident.owner_binding_count;
  result.endpoint_count = proof.residents.count;
  result.stage_count = proof.graph_resident.stage_count;
  result.frame_capacity = proof.graph_wavefront.frame_capacity;
  result.batch_count = proof.graph_wavefront.batch_count;
  result.native_submits = native.native_submit_count;
  result.dispatches = native.payload_dispatch_count;
  result.finals = native.final_callback_count;
  result.epoch_submits = native.epoch_native_submit_count;
  result.pages = native.page_count;
  result.wavefront_steps = native.graph_wavefront_steps;
  result.gpu_read_bytes = native.gpu_backing_read_bytes;
  result.gpu_write_bytes = native.gpu_backing_write_bytes;
  result.generated_pages = native.generated_epochs;
  result.forecasted_pages = native.forecasted_pages;
  result.promoted_pages = native.promoted_pages;
  result.completed_pages = native.completed_epochs;
  result.drained_pages = native.drained_pages;
  result.persisted_pages = native.persisted_pages;
  result.hash_observations = owner->evidence->output_hash_observation_count;
  result.hash_reuses = owner->evidence->output_hash_reuse_count;
  result.host_turns = native.host_service_turn_count;
  result.host_callbacks = native.host_epoch_callback_count;
  result.quarantined = owner->evidence->quarantined;
  result.ok = result.ok && result.graph_resident && result.proof_valid &&
              result.alias_reuse && native.may_write;
  return result;
}

} // namespace rund::measure::compute::virtual_graph_residency::run_detail
