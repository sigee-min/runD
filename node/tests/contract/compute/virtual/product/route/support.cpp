#include "internal.hpp"

#include <algorithm>
#include <cstring>

namespace rund_node_test_virtual::product::route_detail {

std::mutex protocol_mutex;
std::condition_variable protocol_cv;
ObserverProtocol protocol;

[[nodiscard]] bool incompatible_owners(const std::uint32_t existing,
                                       const std::uint32_t incoming) noexcept {
  if ((existing & incoming) != 0u) {
    return false;
  }
  switch (incoming) {
  case OwnerAccelRolling:
    return (existing &
            (OwnerWindow | OwnerDeviceVsm | OwnerServiceFreeDirect)) != 0u;
  case OwnerPersistent:
    return (existing &
            (OwnerWindow | OwnerDeviceVsm | OwnerServiceFreeDirect)) != 0u;
  case OwnerWindow:
    return existing != 0u;
  case OwnerDeviceVsm:
    return (existing &
            (OwnerWindow | OwnerPersistent | OwnerServiceFreeDirect)) != 0u;
  case OwnerServiceFreeDirect:
    return existing != 0u;
  }
  return true;
}

void record_owner(ProductRouteObservation *const observation,
                  const std::uint32_t owner) noexcept {
  if (observation == nullptr) {
    return;
  }
  std::lock_guard lock{protocol_mutex};
  if (protocol.observation != observation || protocol.stopping) {
    return;
  }
  if (incompatible_owners(observation->accepted_owner_mask, owner)) {
    ++observation->conflict_count;
  }
  observation->accepted_owner_mask |= owner;
  ++observation->accepted_owner_count;
}

void record_sliding_prepare(ProductRouteObservation *const observation,
                            const Status status) noexcept {
  if (observation == nullptr) {
    return;
  }
  std::lock_guard lock{protocol_mutex};
  if (protocol.observation == observation && !protocol.stopping) {
    observation->sliding_prepare_called = true;
    observation->sliding_prepare_status = status;
  }
}

void record_status(ProductRouteObservation *const observation,
                   Status ProductRouteObservation::*const field,
                   const Status status) noexcept {
  if (observation == nullptr) {
    return;
  }
  std::lock_guard lock{protocol_mutex};
  if (protocol.observation == observation && !protocol.stopping) {
    observation->*field = status;
  }
}

void record_device_vsm_prepare(ProductRouteObservation *const observation,
                               const Status status,
                               const char *const reason) noexcept {
  if (observation == nullptr) {
    return;
  }
  std::lock_guard lock{protocol_mutex};
  if (protocol.observation == observation && !protocol.stopping) {
    observation->device_vsm_prepare_called = true;
    observation->device_vsm_prepare_status = status;
    observation->device_vsm_prepare_reason = reason;
  }
}

void record_flag(ProductRouteObservation *const observation,
                 bool &flag) noexcept {
  if (observation == nullptr) {
    return;
  }
  std::lock_guard lock{protocol_mutex};
  if (protocol.observation == observation && !protocol.stopping) {
    flag = true;
  }
}

void record_accepted(ProductRouteObservation *const observation, bool &called,
                     bool &accepted, const bool value,
                     const std::uint32_t owner) noexcept {
  if (observation == nullptr) {
    return;
  }
  std::lock_guard lock{protocol_mutex};
  if (protocol.observation != observation || protocol.stopping) {
    return;
  }
  called = true;
  accepted = value;
  if (value && owner != 0u) {
    if (incompatible_owners(observation->accepted_owner_mask, owner)) {
      ++observation->conflict_count;
    }
    observation->accepted_owner_mask |= owner;
    ++observation->accepted_owner_count;
  }
}

void record_nonowning_accept(ProductRouteObservation *const observation,
                             bool &called, bool &accepted,
                             const bool value) noexcept {
  if (observation == nullptr) {
    return;
  }
  std::lock_guard lock{protocol_mutex};
  if (protocol.observation == observation && !protocol.stopping) {
    called = true;
    accepted = value;
  }
}

void record_residency_failure(
    ProductRouteObservation *const observation, const rund::AccelCheck result,
    const rund::node::accel::detail::PreparedKernelPipeline
        &prepared) noexcept {
  if (observation == nullptr || result.ok) {
    return;
  }
  std::lock_guard lock{protocol_mutex};
  if (protocol.observation != observation || protocol.stopping ||
      observation->residency_submit_failed) {
    return;
  }
  observation->residency_submit_failed = true;
  observation->residency_submit_owner_valid = true;
  observation->residency_submit_owner = prepared;
  const char *const reason =
      result.reason == nullptr ? "route_observer_invalid" : result.reason;
  const std::size_t length = std::min(
      std::strlen(reason), observation->residency_submit_reason.size() - 1u);
  std::memcpy(observation->residency_submit_reason.data(), reason, length);
  observation->residency_submit_reason[length] = '\0';
  observation->residency_submit_reason_length =
      static_cast<std::uint32_t>(length);
}

[[nodiscard]] bool enter_protocol(std::unique_lock<std::mutex> &lock,
                                   DeviceState &device,
                                   ProductRouteObservation *const observation,
                                   const DeviceOps *&previous,
                                   ProductRouteObservation *&previous_observation,
                                   const DeviceOps *&previous_forward,
                                   bool &outer) noexcept {
  for (;;) {
    if (protocol.device == nullptr) {
      protocol.device = &device;
      protocol.owner = std::this_thread::get_id();
      protocol.forward = device.ops;
      protocol.observation = nullptr;
      protocol.depth = 0u;
      protocol.inflight = 0u;
      protocol.stopping = false;
      outer = true;
    } else if (protocol.device == &device &&
               protocol.owner == std::this_thread::get_id()) {
      if (protocol.stopping ||
          (observation != nullptr && protocol.observation != nullptr)) {
        return false;
      }
      outer = false;
    } else {
      protocol_cv.wait(lock, [] { return protocol.device == nullptr; });
      continue;
    }
    previous = device.ops;
    previous_observation = protocol.observation;
    previous_forward = protocol.forward;
    return true;
  }
}

void finish_protocol(DeviceState &device, const DeviceOps *const previous,
                     ProductRouteObservation *const previous_observation,
                     const DeviceOps *const previous_forward,
                     const DeviceOps *const installed,
                     const bool outer) noexcept {
  bool wait_for_callbacks = false;
  {
    std::unique_lock lock{protocol_mutex};
    if (protocol.device != &device) {
      return;
    }
    if (protocol.depth != 0u) {
      --protocol.depth;
    }
    if (protocol.inflight != 0u) {
      --protocol.inflight;
    }
    wait_for_callbacks = outer;
    if (wait_for_callbacks) {
      protocol.stopping = true;
    } else if (device.ops == installed) {
      device.ops = previous;
      protocol.observation = previous_observation;
      protocol.forward = previous_forward;
    }
    protocol_cv.notify_all();
  }
  if (!wait_for_callbacks) {
    return;
  }
  std::unique_lock lock{protocol_mutex};
  protocol_cv.wait(
      lock, [] { return protocol.depth == 0u && protocol.inflight == 0u; });
  device.ops = previous;
  protocol.observation = previous_observation;
  protocol.forward = previous_forward;
  protocol = {};
  lock.unlock();
  protocol_cv.notify_all();
}

CallbackLease::~CallbackLease() {
  if (!active) {
    return;
  }
  std::lock_guard lock{protocol_mutex};
  if (protocol.inflight != 0u) {
    --protocol.inflight;
  }
  protocol_cv.notify_all();
}

[[nodiscard]] bool begin_callback(DeviceState *const device,
                                  CallbackLease &lease) noexcept {
  std::lock_guard lock{protocol_mutex};
  if (device == nullptr || protocol.device != device || protocol.stopping ||
      protocol.observation == nullptr || protocol.forward == nullptr) {
    return false;
  }
  lease.forward = protocol.forward;
  lease.observation = protocol.observation;
  lease.active = true;
  ++protocol.inflight;
  return true;
}

[[nodiscard]] DeviceState *pipeline_device(
    const VirtualPipelineState &state) noexcept {
  return state.pipeline == nullptr || state.pipeline->device == nullptr
             ? nullptr
             : state.pipeline->device.get();
}

} // namespace rund_node_test_virtual::product::route_detail
