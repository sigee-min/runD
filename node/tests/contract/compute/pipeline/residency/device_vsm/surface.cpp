#include "local.hpp"

#include <concepts>

namespace rund_node_test_pipeline_residency::device_vsm_test {
namespace {

template <typename Request>
concept HasSignalReady = requires(Request request) { request.signal_ready; };
template <typename Request>
concept HasWaitDone = requires(Request request) { request.wait_done; };
template <typename Request>
concept HasAckDone = requires(Request request) { request.ack_done; };
template <typename Request>
concept HasFailService = requires(Request request) { request.fail_service; };
template <typename Request>
concept HasProject = requires(Request request) { request.project; };
template <typename Request>
concept HasRelease = requires(Request request) { request.release; };
template <typename Request>
concept HasReturned = requires(Request request) { request.returned; };
template <typename Request>
concept HasWake = requires(Request request) { request.wake; };

static_assert(!HasSignalReady<accel::DeviceVsmRequest>);
static_assert(!HasWaitDone<accel::DeviceVsmRequest>);
static_assert(!HasAckDone<accel::DeviceVsmRequest>);
static_assert(!HasFailService<accel::DeviceVsmRequest>);
static_assert(!HasProject<accel::DeviceVsmRequest>);
static_assert(!HasRelease<accel::DeviceVsmRequest>);
static_assert(!HasReturned<accel::DeviceVsmRequest>);
static_assert(!HasWake<accel::DeviceVsmRequest>);

} // namespace

bool CheckSurface() noexcept {
  return sizeof(accel::DeviceVsmRequest) == sizeof(accel::DeviceVsmRequest) &&
         sizeof(accel::DeviceVsmProof) == sizeof(accel::DeviceVsmProof);
}

} // namespace rund_node_test_pipeline_residency::device_vsm_test
