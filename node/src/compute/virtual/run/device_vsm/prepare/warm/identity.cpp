#include "local.hpp"

#include "../../../../../type.hpp"

namespace rund::compute::detail::device_vsm_product_detail::warm_detail {
namespace {

[[nodiscard]] bool
same_ring_plan(const VirtualRunProjection &run, const bool expected,
               const node::accel::detail::DeviceVsmProof &proof) noexcept {
  const bool cached =
      proof.topology == node::accel::detail::DeviceVsmTopology::Window &&
      proof.window.ring.gpu_owned;
  if (expected != cached) {
    return false;
  }
  if (!expected) {
    return true;
  }
  const std::size_t element_bytes = type_bytes(run.input_type);
  if (element_bytes == 0u || run.active.input_bytes % element_bytes != 0u ||
      run.input_page_bytes % element_bytes != 0u ||
      run.input_prefix_bytes % element_bytes != 0u) {
    return false;
  }
  const node::accel::detail::DeviceVsmPageGeometry geometry{
      .logical_bytes = run.active.input_bytes,
      .payload_bytes = run.input_payload_bytes,
      .frame_bytes = run.input_page_bytes,
      .read_prefix_bytes = run.input_prefix_bytes,
      .target_offset_bytes = run.input_prefix_bytes,
      .read_suffix_bytes = run.input_page_bytes - run.input_prefix_bytes -
                           run.input_payload_bytes,
      .page_count = page_count(run),
      .element_bytes = static_cast<std::uint32_t>(element_bytes),
  };
  node::accel::detail::DeviceVsmWindowRingPlan plan{};
  return node::accel::detail::device_vsm_window_ring_plan_expected(geometry,
                                                                   plan) &&
         proof.window.ring == plan;
}

[[nodiscard]] node::accel::detail::DeviceVsmTopology
topology(const VirtualPipelineState &state, const VirtualRunProjection &run,
         const VirtualDeviceVsmRouteProof &proof) noexcept {
  if (run.graph_reduction()) {
    return node::accel::detail::DeviceVsmTopology::GraphMapReduce;
  }
  if (run.graph_execution()) {
    return proof.kind() == VirtualDeviceVsmRouteKind::GraphResident
               ? node::accel::detail::DeviceVsmTopology::GraphResident
               : node::accel::detail::DeviceVsmTopology::GraphPointwise;
  }
  if (run.scan()) {
    return node::accel::detail::DeviceVsmTopology::Scan;
  }
  if (run.reduction()) {
    return node::accel::detail::DeviceVsmTopology::Reduce;
  }
  return state.geometry.route == VirtualRoute::Window
             ? node::accel::detail::DeviceVsmTopology::Window
             : node::accel::detail::DeviceVsmTopology::Pointwise;
}

} // namespace

std::uint64_t page_count(const VirtualRunProjection &run) noexcept {
  return run.graph_execution() ? run.active.graph.page_count()
                               : run.active.stream.page_count();
}

bool owner_ready(const DeviceVsmProductOwner *const owner) noexcept {
  return owner != nullptr && owner->proof != nullptr && owner->preparation &&
         owner->memory != nullptr && owner->registration != nullptr &&
         owner->release_registration != nullptr && owner->evidence != nullptr;
}

bool owner_shape_matches(const VirtualPipelineState &state,
                         const VirtualRunProjection &run,
                         const node::accel::detail::DeviceVsmIdentity &identity,
                         const VirtualDeviceVsmRouteProof &proof,
                         const DeviceVsmProductOwner &owner) noexcept {
  return owner.proof != nullptr && owner.proof->identity == identity &&
         owner.route_proof == proof &&
         owner.proof->topology == topology(state, run, proof) &&
         owner.proof->geometry.page_count == page_count(run) &&
         owner.proof->geometry.logical_bytes == run.active.input_bytes &&
         owner.proof->output_bytes == run.active.output_bytes &&
         owner.input_count == run.input_count &&
         owner.proof->residents.input_count == run.input_count &&
         same_ring_plan(run,
                        proof.kind() == VirtualDeviceVsmRouteKind::WindowRing,
                        *owner.proof) &&
         resident_bindings_match(state, run, owner);
}

} // namespace rund::compute::detail::device_vsm_product_detail::warm_detail
