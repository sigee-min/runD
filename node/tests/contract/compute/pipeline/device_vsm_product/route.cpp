#include "route.hpp"
#include "../../virtual/product/route/proof.hpp"

#include "src/compute/backend.hpp"
#include "src/compute/device/state.hpp"
#include "src/compute/virtual/run/device_vsm.hpp"
#include "src/compute/virtual/run/device_vsm/operations.hpp"
#include "src/compute/virtual/state.hpp"

namespace rund_node_test_device_vsm_product {
namespace {

using rund::compute::Stats;
using rund::compute::Status;
using rund::compute::VirtualBacking;
using rund::compute::detail::VirtualDeviceVsmRouteProof;
using rund::compute::detail::VirtualExecutionDeviceVsmPrepared;
using rund::compute::detail::VirtualExecutionResult;
using rund::compute::detail::VirtualPipelineState;
using rund::compute::detail::VirtualRunProjection;
using rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner;

[[nodiscard]] bool
proof_tamper_rejected(const VirtualPipelineState &state,
                      const VirtualRunProjection &run,
                      const VirtualDeviceVsmRouteProof &proof) noexcept {
  using rund::compute::detail::device_vsm_product_detail::proof_matches;
  if (!proof.valid() || !proof_matches(state, run, proof)) {
    return false;
  }
  if (!rund_node_test_virtual::product::route_detail::proof_tamper_rejected(
          proof)) {
    return false;
  }
  VirtualDeviceVsmRouteProof stamped = proof;
  stamped.stamp_hi ^= 1u;
  if (proof_matches(state, run, stamped))
    return false;
  if (const auto *window = proof.window()) {
    auto changed = *window;
    changed.input_bytes ^= 1u;
    const VirtualDeviceVsmRouteProof snapshot{
        rund::compute::detail::VirtualDeviceVsmWindowRing{changed},
        proof.stamp_hi, proof.stamp_lo};
    return snapshot.valid() && !proof_matches(state, run, snapshot);
  }
  return true;
}

thread_local RouteObservation *active_observation{};
thread_local decltype(rund::compute::detail::VirtualDeviceOps::
                          prepare_virtual_device_vsm_product) active_prepare{};
thread_local decltype(rund::compute::detail::VirtualDeviceOps::
                          execute_virtual_device_vsm_product) active_execute{};

Status PrepareObserved(VirtualPipelineState &state,
                       const VirtualRunProjection &run,
                       const VirtualDeviceVsmRouteProof proof,
                       VirtualExecutionDeviceVsmPrepared &prepared) noexcept {
  if (!proof_tamper_rejected(state, run, proof)) {
    return Status::fail(rund::compute::Reason::PipelineInvalid);
  }
  const Status status = active_prepare(state, run, proof, prepared);
  if (active_observation != nullptr) {
    active_observation->prepare_reason = prepared.reason;
  }
  if (status && active_observation != nullptr) {
    active_observation->owner = prepared.owner;
    active_observation->prepared = true;
  }
  return status;
}

VirtualExecutionResult ExecuteObserved(
    VirtualPipelineState &state, const std::span<VirtualBacking *const> inputs,
    VirtualBacking &output, const VirtualRunProjection &run,
    const VirtualExecutionDeviceVsmPrepared &prepared, Stats &stats) noexcept {
  const VirtualExecutionResult result =
      active_execute(state, inputs, output, run, prepared, stats);
  if (active_observation == nullptr) {
    return result;
  }
  active_observation->executed = true;
  active_observation->execution_status = static_cast<bool>(result.status);
  active_observation->execution_poison = result.poison_pipeline;
  active_observation->execution_reason =
      static_cast<unsigned>(result.status.reason());
  const auto owner =
      std::static_pointer_cast<DeviceVsmProductOwner>(prepared.owner);
  if (owner != nullptr && owner->evidence != nullptr) {
    active_observation->evidence = *owner->evidence;
  }
  return result;
}

class RouteScope final {
public:
  RouteScope(rund::compute::detail::DeviceState &device,
             RouteObservation &observation) noexcept
      : device_(device), original_(device.ops) {
    if (original_ == nullptr || active_observation != nullptr ||
        original_->virtual_execution.prepare_virtual_device_vsm_product ==
            nullptr ||
        original_->virtual_execution.execute_virtual_device_vsm_product ==
            nullptr) {
      return;
    }
    routed_ = *original_;
    active_prepare =
        original_->virtual_execution.prepare_virtual_device_vsm_product;
    active_execute =
        original_->virtual_execution.execute_virtual_device_vsm_product;
    routed_.virtual_execution.prepare_virtual_device_vsm_product =
        PrepareObserved;
    routed_.virtual_execution.execute_virtual_device_vsm_product =
        ExecuteObserved;
    active_observation = &observation;
    observation.production_route = true;
    device_.ops = &routed_;
    active_ = true;
  }

  RouteScope(const RouteScope &) = delete;
  RouteScope &operator=(const RouteScope &) = delete;

  ~RouteScope() {
    if (!active_) {
      return;
    }
    device_.ops = original_;
    active_observation = nullptr;
    active_prepare = nullptr;
    active_execute = nullptr;
  }

  [[nodiscard]] explicit operator bool() const noexcept { return active_; }

private:
  rund::compute::detail::DeviceState &device_;
  const rund::compute::detail::DeviceOps *original_{};
  rund::compute::detail::DeviceOps routed_{};
  bool active_{};
};

} // namespace

Status RunThroughDeviceVsmProductRoute(
    const std::shared_ptr<VirtualPipelineState> &state,
    RouteObservation &observation, const bool force_device_vsm) noexcept {
  observation = {};
  if (state == nullptr || state->pipeline == nullptr ||
      state->pipeline->device == nullptr) {
    return Status::fail(rund::compute::Reason::PipelineInvalid);
  }
  RouteScope route{*state->pipeline->device, observation};
  if (!route) {
    return Status::fail(rund::compute::Reason::PipelineBusy);
  }
  // This harness is the explicit DeviceVsm contract. Ordinary callback-backed
  // unary runs deliberately decline DeviceVsm so the public route can reach
  // Persistent fixed-W service; this marker keeps direct DeviceVsm evidence
  // and fault tests intentional without widening the production default.
  const bool required = state->geometry.device_vsm_required;
  if (force_device_vsm) {
    state->geometry.device_vsm_required = true;
  }
  const Status status = rund::compute::detail::run_virtual_pipeline(state);
  if (force_device_vsm) {
    state->geometry.device_vsm_required = required;
  }
  return status;
}

Status RunThroughDeviceVsmProductRoute(
    const std::shared_ptr<VirtualPipelineState> &state,
    RouteObservation &observation) noexcept {
  return RunThroughDeviceVsmProductRoute(state, observation, true);
}

} // namespace rund_node_test_device_vsm_product
