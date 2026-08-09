#include "fault.hpp"

#include "../backend/token.hpp"

#include <memory>

namespace rund::node::accel::detail {

bool InjectNativeDeviceLostOnce(const rund::AccelDevice &pick) noexcept {
  const std::shared_ptr<PickToken> token = AdmitPick(pick);
  return token != nullptr && token->ops != nullptr &&
         token->ops->inject_device_lost_once != nullptr &&
         token->ops->inject_device_lost_once(token->raw);
}

bool InjectNativeTraceUnavailableOnce(const rund::AccelDevice &pick) noexcept {
  const std::shared_ptr<PickToken> token = AdmitPick(pick);
  return token != nullptr && token->ops != nullptr &&
         token->ops->inject_trace_unavailable_once != nullptr &&
         token->ops->inject_trace_unavailable_once(token->raw);
}

bool InjectNativeTraceResolveDeviceLostOnce(
    const rund::AccelDevice &pick) noexcept {
  const std::shared_ptr<PickToken> token = AdmitPick(pick);
  return token != nullptr && token->ops != nullptr &&
         token->ops->inject_trace_resolve_device_lost_once != nullptr &&
         token->ops->inject_trace_resolve_device_lost_once(token->raw);
}

} // namespace rund::node::accel::detail
