#include <accel/context/value.hpp>
#include <accel/device.hpp>

#include "local.hpp"

namespace rund::node::accel::detail {

ContextTokenAdmission AdmitContextToken(const rund::AccelContext &context) {
  std::shared_ptr<ContextToken> token = LookupContextToken(context.owner);
  if (token == nullptr || !SameObject(token, context.owner) ||
      token->pick == nullptr || token->pick->ops == nullptr ||
      !token->pick->ops->resident || !context.check.ok ||
      context.id != token->id || context.api != token->api ||
      !SameCaps(context.caps, token->caps) || !context.evidence.ok ||
      context.evidence.api != token->api ||
      !SameCaps(context.evidence.caps, token->caps) ||
      !SameReason(context.evidence.reason, "ok")) {
    return ContextTokenAdmission{
        .check = RejectAccelCheck("accel_context_buffer_invalid")};
  }

  const rund::AccelDevice &raw = token->pick->raw;
  if (!SameCheck(context.pick.check, raw.check) ||
      context.pick.api != raw.api || context.pick.owner == nullptr ||
      !context.pick.caps.ok || !SameObject(context.pick.owner, token->pick) ||
      !SameCaps(context.pick.caps, raw.caps) ||
      !SameCpuCaps(context.pick.cpu_caps, raw.cpu_caps) ||
      !SameDispatch(context.pick.backend, raw.backend) ||
      !SameBackendInfo(context.pick.backend_info, raw.backend_info)) {
    return ContextTokenAdmission{
        .check = RejectAccelCheck("accel_context_buffer_invalid")};
  }

  return ContextTokenAdmission{
      .check = OkAccelCheck(),
      .token = token,
  };
}

} // namespace rund::node::accel::detail
