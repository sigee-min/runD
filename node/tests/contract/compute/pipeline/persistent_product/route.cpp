#include "route.hpp"

#include "backing.hpp"

#include "src/compute/backend.hpp"
#include "src/compute/device/state.hpp"
#include "src/compute/virtual/run/sliding.hpp"
#include "src/compute/virtual/run/sliding/internal.hpp"
#include "src/compute/virtual/state.hpp"

#include <mutex>

namespace rund_node_test_persistent_product {
namespace {

using rund::compute::Stats;
using rund::compute::Status;
using rund::compute::VirtualBacking;
using rund::compute::detail::VirtualExecutionResult;
using rund::compute::detail::VirtualExecutionSlidingPrepared;
using rund::compute::detail::VirtualPipelineState;
using rund::compute::detail::VirtualRunProjection;

thread_local ProductRouteObservation *active_observation{};
thread_local decltype(rund::compute::detail::VirtualDeviceOps::
                          prepare_virtual_sliding_product) active_prepare{};
thread_local decltype(rund::compute::detail::VirtualDeviceOps::
                          execute_virtual_sliding_product) active_execute{};

Status PrepareObserved(VirtualPipelineState &state,
                       const VirtualRunProjection &run,
                       VirtualExecutionSlidingPrepared &prepared) noexcept {
  const Status status =
      active_prepare == nullptr
          ? rund::compute::detail::prepare_accel_virtual_execution_sliding(
                state, run, prepared)
          : active_prepare(state, run, prepared);
  if (active_observation != nullptr) {
    active_observation->prepare_status = status;
    active_observation->prepared = static_cast<bool>(prepared);
    if (status) {
      active_observation->owner = prepared.owner;
      const auto owner = std::static_pointer_cast<
          rund::compute::detail::sliding_product_detail::SlidingProductOwner>(
          prepared.owner);
      if (owner != nullptr) {
        active_observation->prepared_mode = owner->mode;
      }
    }
  }
  return status;
}

VirtualExecutionResult
ExecuteObserved(VirtualPipelineState &state, VirtualBacking &input,
                VirtualBacking &output, const VirtualRunProjection &run,
                const VirtualExecutionSlidingPrepared &prepared,
                Stats &stats) noexcept {
  if (active_observation != nullptr &&
      active_observation->fail_input != nullptr) {
    active_observation->fail_input->fail_next_read();
    active_observation->fail_input = nullptr;
  }
  if (active_observation != nullptr &&
      active_observation->force_terminal_unsupported) {
    return VirtualExecutionResult{
        .status = Status::fail(rund::compute::Reason::BackendUnsupported)};
  }
  const VirtualExecutionResult result =
      active_execute == nullptr
          ? rund::compute::detail::execute_accel_virtual_execution_sliding(
                state, input, output, run, prepared, stats)
          : active_execute(state, input, output, run, prepared, stats);
  if (active_observation == nullptr || active_observation->owner == nullptr) {
    return result;
  }
  const auto owner = std::static_pointer_cast<
      rund::compute::detail::sliding_product_detail::SlidingProductOwner>(
      active_observation->owner);
  if (owner == nullptr || owner->run == nullptr) {
    return result;
  }
  std::lock_guard lock{owner->run->gate};
  const auto request = owner->run->persistent_preparation.active_request();
  active_observation->request_mode = request.mode;
  active_observation->capability_mode =
      owner->run->persistent_preparation.backend.capability.mode;
  {
    std::lock_guard control_lock{owner->run->persistent_control.gate};
    active_observation->control_mode = owner->run->persistent_control.mode;
  }
  active_observation->mode_snapshot = true;
  active_observation->final_received = owner->run->persistent_final_received;
  if (active_observation->final_received) {
    active_observation->final = owner->run->persistent_final;
  }
  return result;
}

class RouteScope final {
public:
  RouteScope(rund::compute::detail::DeviceState &device,
             ProductRouteObservation &observation) noexcept
      : device_(device), original_(device.ops) {
    if (original_ == nullptr || active_observation != nullptr ||
        (original_->virtual_execution.prepare_virtual_sliding_product ==
         nullptr) !=
            (original_->virtual_execution.execute_virtual_sliding_product ==
             nullptr)) {
      return;
    }
    routed_ = *original_;
    active_prepare =
        original_->virtual_execution.prepare_virtual_sliding_product;
    active_execute =
        original_->virtual_execution.execute_virtual_sliding_product;
    routed_.virtual_execution.prepare_virtual_device_vsm_product = nullptr;
    routed_.virtual_execution.execute_virtual_device_vsm_product = nullptr;
    routed_.virtual_execution.prepare_virtual_sliding_product = PrepareObserved;
    routed_.virtual_execution.execute_virtual_sliding_product = ExecuteObserved;
    active_observation = &observation;
    observation.production_route = active_prepare != nullptr;
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

Status RunThroughPersistentProductRoute(
    const std::shared_ptr<VirtualPipelineState> &state,
    ProductRouteObservation &observation) noexcept {
  PersistentProductBacking *const fail_input = observation.fail_input;
  const bool force_terminal_unsupported =
      observation.force_terminal_unsupported;
  observation = {};
  observation.fail_input = fail_input;
  observation.force_terminal_unsupported = force_terminal_unsupported;
  if (state == nullptr || state->pipeline == nullptr ||
      state->pipeline->device == nullptr) {
    return Status::fail(rund::compute::Reason::PipelineInvalid);
  }
  RouteScope route{*state->pipeline->device, observation};
  return route ? rund::compute::detail::run_virtual_pipeline(state)
               : Status::fail(rund::compute::Reason::PipelineBusy);
}

} // namespace rund_node_test_persistent_product
