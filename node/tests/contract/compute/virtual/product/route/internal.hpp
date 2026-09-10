#pragma once

#include "proof.hpp"

#include "../model.hpp"
#include "../route.hpp"

#include "src/compute/virtual/state.hpp"
#include "src/compute/virtual/run/execution.hpp"
#include "src/compute/virtual/run/sliding.hpp"

#include <rund/compute/pipeline.hpp>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <thread>

namespace rund_node_test_virtual::product::route_detail {

using rund::compute::Stats;
using rund::compute::Status;
using rund::compute::VirtualBacking;
using rund::compute::detail::DeviceOps;
using rund::compute::detail::DeviceState;
using rund::compute::detail::VirtualDeviceVsmRouteProof;
using rund::compute::detail::VirtualExecutionDeviceVsmPrepared;
using rund::compute::detail::VirtualExecutionResult;
using rund::compute::detail::VirtualExecutionSlidingPrepared;
using rund::compute::detail::VirtualPipelineState;
using rund::compute::detail::VirtualRunProjection;

struct ObserverProtocol final {
  DeviceState *device{};
  std::thread::id owner{};
  const DeviceOps *forward{};
  ProductRouteObservation *observation{};
  std::size_t depth{};
  std::size_t inflight{};
  bool stopping{};
};

extern std::mutex protocol_mutex;
extern std::condition_variable protocol_cv;
extern ObserverProtocol protocol;

[[nodiscard]] bool incompatible_owners(std::uint32_t existing,
                                       std::uint32_t incoming) noexcept;

void record_owner(ProductRouteObservation *, std::uint32_t owner) noexcept;
void record_sliding_prepare(ProductRouteObservation *, Status) noexcept;
void record_status(ProductRouteObservation *,
                   Status ProductRouteObservation::*, Status) noexcept;
void record_device_vsm_prepare(ProductRouteObservation *, Status,
                               const char *reason) noexcept;
void record_flag(ProductRouteObservation *, bool &) noexcept;
void record_accepted(ProductRouteObservation *, bool &, bool &, bool,
                     std::uint32_t owner) noexcept;
void record_nonowning_accept(ProductRouteObservation *, bool &, bool &,
                             bool) noexcept;
void record_residency_failure(
    ProductRouteObservation *, rund::AccelCheck,
    const rund::node::accel::detail::PreparedKernelPipeline &) noexcept;

[[nodiscard]] bool enter_protocol(
    std::unique_lock<std::mutex> &, DeviceState &, ProductRouteObservation *,
    const DeviceOps *&, ProductRouteObservation *&, const DeviceOps *&,
    bool &outer) noexcept;
void finish_protocol(DeviceState &, const DeviceOps *,
                     ProductRouteObservation *, const DeviceOps *,
                     const DeviceOps *, bool outer) noexcept;

struct CallbackLease final {
  const DeviceOps *forward{};
  ProductRouteObservation *observation{};
  bool active{};

  CallbackLease() = default;
  CallbackLease(const CallbackLease &) = delete;
  CallbackLease &operator=(const CallbackLease &) = delete;

  ~CallbackLease();
};

[[nodiscard]] bool begin_callback(DeviceState *, CallbackLease &) noexcept;
[[nodiscard]] DeviceState *pipeline_device(
    const VirtualPipelineState &) noexcept;

Status ObserveSlidingPrepare(VirtualPipelineState &, const VirtualRunProjection &,
                             VirtualExecutionSlidingPrepared &) noexcept;
VirtualExecutionResult ObserveSlidingExecute(
    VirtualPipelineState &, VirtualBacking &, VirtualBacking &,
    const VirtualRunProjection &, const VirtualExecutionSlidingPrepared &,
    Stats &) noexcept;
Status ObserveDeviceVsmPrepare(VirtualPipelineState &, const VirtualRunProjection &,
                               VirtualDeviceVsmRouteProof,
                               VirtualExecutionDeviceVsmPrepared &) noexcept;
VirtualExecutionResult ObserveDeviceVsmExecute(
    VirtualPipelineState &, std::span<VirtualBacking *const>, VirtualBacking &,
    const VirtualRunProjection &, const VirtualExecutionDeviceVsmPrepared &,
    Stats &) noexcept;
Status ObserveServiceFreeDirect(
    const std::shared_ptr<rund::compute::detail::PipelineState> &, bool &) noexcept;
rund::AccelCheck ObserveResidencyPipeline(
    const DeviceState &,
    const rund::node::accel::detail::PreparedKernelPipeline &,
    std::span<const std::uint32_t>, std::shared_ptr<void>,
    rund::node::accel::detail::PreparedPipelineCompletion, void *) noexcept;
Status ObserveResidencyWindow(
    const DeviceState &,
    const rund::node::accel::detail::PreparedResidencyWindowRequest &,
    rund::node::accel::detail::PreparedResidencyWindowControl &) noexcept;
Status ObserveResidencyStreamWindow(
    const DeviceState &,
    const rund::node::accel::detail::PreparedResidencyWindowRequest &,
    rund::node::accel::detail::PreparedResidencyWindowControl &,
    rund::node::accel::detail::PreparedResidencyStreamControl &) noexcept;
Status ObserveResidencySchedule(
    const DeviceState &,
    const rund::node::accel::detail::PreparedResidencyScheduleRequest &,
    rund::node::accel::detail::PreparedResidencyScheduleControl &,
    rund::node::accel::detail::PreparedResidencyStreamControl &) noexcept;
Status ObservePrepareResidencySliding(
    const DeviceState &,
    std::span<const rund::node::accel::detail::PreparedResidencySlidingRole>,
    rund::node::accel::detail::ResidencySlidingMemory,
    rund::node::accel::detail::PreparedResidencySlidingControl &) noexcept;
Status ObserveSubmitResidencySliding(
    const DeviceState &,
    const rund::node::accel::detail::PreparedResidencySlidingRequest &,
    rund::node::accel::detail::PreparedResidencySlidingControl &) noexcept;

} // namespace rund_node_test_virtual::product::route_detail
