#include "internal.hpp"

#include <rund/compute/pipeline/runtime.hpp>

namespace rund::compute::detail::device_vsm_product_detail {

node::accel::detail::DeviceVsmIdentity
route_stamp(const VirtualPipelineState &state, const VirtualRunProjection &run,
            const VirtualDeviceVsmRouteProof &proof) noexcept {
  const std::shared_ptr<PipelineState> selected =
      run.graph_reduction && state.device_vsm_semantic_pipeline != nullptr
          ? state.device_vsm_semantic_pipeline
          : state.pipeline;
  const graph::Fingerprint pipeline = pipeline_fingerprint(selected);
  const std::shared_ptr<PipelineState> terminal_pipeline =
      run.graph_execution ? graph_terminal_pipeline(state, 0u) : nullptr;
  const graph::Fingerprint collective =
      run.graph_execution ? pipeline_fingerprint(terminal_pipeline)
                          : graph::Fingerprint{};
  ::rund::node::hash_detail::Fnv hi{
      ::rund::node::hash_detail::kFnvStandardOffset};
  ::rund::node::hash_detail::Fnv lo{};
  constexpr char domain[] = "rund.compute.virtual.device-vsm.v4";
  hi.Bytes(reinterpret_cast<const std::uint8_t *>(domain), sizeof(domain) - 1u);
  lo.Bytes(reinterpret_cast<const std::uint8_t *>(domain), sizeof(domain) - 1u);
  const auto mix = [&](const std::uint64_t value) noexcept {
    hi.Number(value);
    lo.Number(value ^ 0x9e3779b97f4a7c15ull);
  };
  mix(static_cast<std::uint64_t>(proof.kind));
  mix(static_cast<std::uint64_t>(proof.endpoint));
  mix(static_cast<std::uint64_t>(run.input_type));
  mix(static_cast<std::uint64_t>(run.output_type));
  mix(proof.page_count);
  mix(proof.frame_capacity);
  const VirtualWindowPreflight &window = proof.window;
  mix(static_cast<std::uint64_t>(window.mode));
  mix(static_cast<std::uint64_t>(window.endpoint));
  mix(window.page_count);
  mix(window.frame_capacity);
  mix(window.input_bytes);
  mix(window.output_bytes);
  mix(window.input_frame_bytes);
  mix(window.output_frame_bytes);
  mix(window.frame_bytes);
  mix(window.device_storage_bytes);
  mix(window.host.input);
  mix(window.host.output);
  mix(window.host.storage_bytes);
  mix(pipeline.hi);
  mix(pipeline.lo);
  mix(collective.hi);
  mix(collective.lo);
  if (run.graph_execution && !run.graph_reduction) {
    const std::size_t stage_count =
        state.pipeline == nullptr || state.pipeline->residency == nullptr
            ? 0u
            : state.pipeline->residency->tiled_graph().stages().size();
    mix(stage_count);
    for (std::size_t stage = 0u; stage < stage_count; ++stage) {
      const graph::Fingerprint fingerprint =
          pipeline_fingerprint(graph_stage_pipeline(state, stage, 0u));
      mix(fingerprint.hi);
      mix(fingerprint.lo);
    }
    mix(node::accel::detail::DeviceVsmGraphControllerVersion);
    mix(run.frame_capacity);
    mix(run.active.graph.page_count());
  }
  const bool staged_graph =
      proof.kind == VirtualDeviceVsmRouteKind::GraphResident;
  mix(run.input_count);
  for (std::size_t index = 0u; index < run.input_count; ++index) {
    mix(run.inputs[index].identity_hi);
    mix(run.inputs[index].identity_lo);
    mix(run.inputs[index].backing);
    if (staged_graph) {
      mix(run.inputs[index].version);
    }
  }
  mix(run.result_identity_hi);
  mix(run.result_identity_lo);
  mix(run.output_backing);
  mix(run.active.input_bytes);
  mix(run.input_payload_bytes);
  mix(run.input_page_bytes);
  mix(run.output_page_bytes);
  mix(run.input_prefix_bytes);
  mix(run.output_prefix_bytes);
  mix(static_cast<std::uint64_t>(state.geometry.route));
  mix(state.geometry.operation);
  mix(state.geometry.boundary);
  mix(state.geometry.materialization_hi);
  mix(state.geometry.materialization_lo);
  mix(proof.kind == VirtualDeviceVsmRouteKind::WindowRing ? 0x57524e47u : 0u);
  mix(run.graph_execution ? run.active.graph.page_count()
                          : run.active.stream.page_count());
  mix(run.graph_execution && state.pipeline != nullptr &&
              state.pipeline->residency != nullptr
          ? state.pipeline->residency->tiled_graph().stages().size()
          : 0u);
  const residency::Identity residency_identity =
      run.graph_execution && state.pipeline != nullptr &&
              state.pipeline->residency != nullptr
          ? state.pipeline->residency->identity()
          : residency::Identity{};
  mix(residency_identity.hi);
  mix(residency_identity.lo);
  node::accel::detail::DeviceVsmIdentity identity{.hi = hi.Finish(),
                                                  .lo = lo.Finish()};
  if (!identity) {
    identity.lo = 1u;
  }
  return identity;
}

bool proof_matches(const VirtualPipelineState &state,
                   const VirtualRunProjection &run,
                   const VirtualDeviceVsmRouteProof &proof) noexcept {
  if (!proof.valid()) {
    return false;
  }
  VirtualDeviceVsmRouteProof unsigned_proof = proof;
  unsigned_proof.stamp_hi = 0u;
  unsigned_proof.stamp_lo = 0u;
  const node::accel::detail::DeviceVsmIdentity expected =
      route_stamp(state, run, unsigned_proof);
  return proof.stamp_hi == expected.hi && proof.stamp_lo == expected.lo;
}

} // namespace rund::compute::detail::device_vsm_product_detail
