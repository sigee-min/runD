#include "../backend.hpp"
#include "../device/state.hpp"
#include "../status.hpp"

#if defined(RUND_NODE_OPEN_PROBE)
#include "probe.hpp"
#endif

#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <node/accel/pick.hpp>

#include <memory>

namespace rund::compute::detail {
namespace {

[[nodiscard]] rund::AccelPolicy
exact_policy(const rund::AccelApi api) noexcept {
  rund::AccelPolicy result{};
  result.allow_fake = false;
  result.preferred[0] = api;
  result.preferred_count = 1u;
  return result;
}

[[nodiscard]] Result<std::shared_ptr<DeviceState>>
open_accel_config(const Backend requested, const rund::AccelApi api) {
#if defined(RUND_NODE_OPEN_PROBE)
  observe_open_config();
#endif
  try {
    rund::AccelDevice pick = node::accel::PickAccel(exact_policy(api));
    if (!pick.check.ok) {
      return Result<std::shared_ptr<DeviceState>>::fail(
          project_reason(pick.check.reason, Reason::AdapterUnavailable));
    }
    if (pick.api != api) {
      return Result<std::shared_ptr<DeviceState>>::fail(
          Reason::BackendUnsupported);
    }
    rund::AccelContext context = node::accel::OpenAccel(pick);
    if (!context.check.ok) {
      return Result<std::shared_ptr<DeviceState>>::fail(
          project_reason(context.check.reason, Reason::AdapterUnavailable));
    }
    auto state = std::make_shared<DeviceState>();
    state->backend = requested;
    state->storage = AccelDeviceState{
        .pick = std::move(pick),
        .context = std::move(context),
    };
    state->ops = &AccelDeviceOps();
    return Result<std::shared_ptr<DeviceState>>::success(std::move(state));
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<DeviceState>>::fail(Reason::DeviceCapacity);
  }
}

} // namespace

Result<std::shared_ptr<DeviceState>> open_metal(const std::uint32_t workers) {
  (void)workers;
  return open_accel_config(Backend::Metal, rund::AccelApi::Metal);
}

Result<std::shared_ptr<DeviceState>> open_vulkan(const std::uint32_t workers) {
  (void)workers;
  return open_accel_config(Backend::Vulkan, rund::AccelApi::Vulkan);
}

} // namespace rund::compute::detail
