#include "internal.hpp"

#include "../../readiness.hpp"
#include "../internal.hpp"
#include "../operations.hpp"

#include "../../../../backend.hpp"
#include "../../../backing.hpp"
#include "../../../graph/reduce/wavefront.hpp"

#include <array>

namespace rund::compute::detail {
namespace {

[[nodiscard]] bool
defer_ordinary_callback_route(const VirtualPipelineState &state,
                              const std::span<VirtualBacking *const> inputs,
                              const VirtualBacking &output,
                              const VirtualRunProjection &run,
                              const bool graph_resident_accepted) noexcept {
  if (state.pipeline == nullptr || state.pipeline->device == nullptr ||
      state.pipeline->device->backend == Backend::Cpu ||
      run.device_vsm_required || run.scan() || run.multi_pointwise() ||
      run.multi_scan() || run.poolless_device_vsm() ||
      VirtualBackingAccess::resident(output) != nullptr) {
    return false;
  }

  if (run.graph_execution()) {
    if (graph_resident_accepted) {
      return false;
    }
    if (!device_vsm_route_detail::graph_host_product_eligible(state, inputs,
                                                              run)) {
      return false;
    }
    return graph_reduce::graph_wavefront_parallel_eligible(state, run) ||
           VirtualBackingAccess::write_lanes(output) >= 2u;
  }
  if (run.reduction() || run.graph_reduction() || run.input_count != 1u ||
      inputs.size() != 1u || inputs.front() == nullptr ||
      inputs.front()->max_parallel_reads() > 1u ||
      VirtualBackingAccess::resident(*inputs.front()) != nullptr) {
    return false;
  }

  if (run.active.stream.epoch_count() < 2u) {
    return false;
  }

  const bool pointwise = state.geometry.route == VirtualRoute::Pointwise;
  const bool window = state.geometry.route == VirtualRoute::Window;
  return pointwise || window;
}

[[nodiscard]] VirtualDeviceVsmCandidate
unavailable_candidate(const bool required,
                      const ::rund::AccelCheck admission) noexcept {
  return VirtualDeviceVsmCandidate{
      .phase = required ? VirtualDeviceVsmPhase::Terminal
                        : VirtualDeviceVsmPhase::Ordinary,
      .status = required ? Status::fail(Reason::BackendUnsupported)
                         : Status::success(),
      .admission = admission};
}

[[nodiscard]] VirtualDeviceVsmCandidate
unavailable_candidate(const bool required) noexcept {
  return unavailable_candidate(
      required, ::rund::AccelCheck{false, "compute_backend_unsupported"});
}

} // namespace

VirtualDeviceVsmCandidate probe_virtual_device_vsm_route(
    const VirtualPipelineState &state, VirtualBacking &input,
    const VirtualBacking &output, const VirtualRunProjection &run) noexcept {
  std::array<VirtualBacking *, 1u> inputs{&input};
  return probe_virtual_device_vsm_route(state, inputs, output, run);
}

VirtualDeviceVsmCandidate
probe_virtual_device_vsm_route(const VirtualPipelineState &state,
                               const std::span<VirtualBacking *const> inputs,
                               const VirtualBacking &output,
                               const VirtualRunProjection &run) noexcept {
  const bool staged_loop =
      device_vsm_product_detail::staged_loop_shape(state, run);
  if (staged_loop) {
    const Status ready = ready_virtual_residency_window(state);
    if (!ready) {
      if (ready.reason() == Reason::BackendUnsupported) {
        return unavailable_candidate(run.device_vsm_required);
      }
      return VirtualDeviceVsmCandidate{
          .phase = VirtualDeviceVsmPhase::Terminal,
          .status = ready,
          .admission = {false, ready.error().data()}};
    }
  }
  VirtualDeviceVsmEndpoint endpoint = staged_loop
                                          ? VirtualDeviceVsmEndpoint::Staged
                                          : VirtualDeviceVsmEndpoint::Invalid;
  const bool window_ring =
      !staged_loop &&
      device_vsm_product_detail::window_ring_shape(state, run, endpoint);
  const device_vsm_route_detail::GraphResidentDecision graph =
      device_vsm_route_detail::graph_resident_decision(state, inputs, run);
  if (graph.declined()) {
    return unavailable_candidate(run.device_vsm_required);
  }
  if (!staged_loop && !window_ring &&
      defer_ordinary_callback_route(state, inputs, output, run,
                                    graph.accepted())) {
    return unavailable_candidate(run.device_vsm_required);
  }
  const bool graph_resident = graph.accepted();
  if (graph_resident) {
    endpoint =
        graph.endpoint == device_vsm_product_detail::EndpointMode::Resident
            ? VirtualDeviceVsmEndpoint::Resident
            : VirtualDeviceVsmEndpoint::Staged;
  }
  if (state.pipeline == nullptr || state.pipeline->device == nullptr ||
      state.pipeline->device->backend == Backend::Cpu) {
    return unavailable_candidate(run.device_vsm_required);
  }
  const std::uint64_t page_count = run.graph_execution()
                                       ? run.active.graph.page_count()
                                       : run.active.stream.page_count();
  VirtualDeviceVsmRouteProof::Route route =
      VirtualDeviceVsmDirect{page_count, run.frame_capacity};
  if (staged_loop) {
    route = VirtualDeviceVsmStagedLoop{page_count, run.frame_capacity};
  } else if (window_ring) {
    route = VirtualDeviceVsmWindowRing{state.window_preflight};
  } else if (graph_resident) {
    route =
        VirtualDeviceVsmGraphResident{page_count, run.frame_capacity, endpoint};
  }
  VirtualDeviceVsmRouteProof proof{route};
  const node::accel::detail::DeviceVsmIdentity stamp =
      device_vsm_product_detail::route_stamp(state, run, proof);
  proof.stamp_hi = stamp.hi;
  proof.stamp_lo = stamp.lo;
  if (!proof.valid()) {
    return unavailable_candidate(run.device_vsm_required);
  }
  const DeviceOps *const ops = state.pipeline->device->ops;
  if (ops == nullptr ||
      ops->virtual_execution.admit_virtual_device_vsm == nullptr) {
    return unavailable_candidate(run.device_vsm_required);
  }
  const ::rund::AccelCheck admission =
      ops->virtual_execution.admit_virtual_device_vsm(state, run, proof);
  if (!admission.ok) {
    return unavailable_candidate(run.device_vsm_required, admission);
  }
  if (ops->virtual_execution.prepare_virtual_device_vsm_product == nullptr ||
      ops->virtual_execution.execute_virtual_device_vsm_product == nullptr) {
    return unavailable_candidate(run.device_vsm_required);
  }
  return VirtualDeviceVsmCandidate{.phase = VirtualDeviceVsmPhase::DeviceVsm,
                                   .status = Status::success(),
                                   .proof = proof,
                                   .admission = admission};
}

} // namespace rund::compute::detail
