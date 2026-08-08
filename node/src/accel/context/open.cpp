#include <accel/check.hpp>
#include <accel/context/evidence.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <accel/runtime.hpp>

#include "../backend/resource.hpp"
#include "local.hpp"

#include <memory>

namespace rund::node::accel {

rund::AccelContext OpenAccel(const rund::AccelDevice &pick) {
  const std::shared_ptr<detail::PickToken> pick_token = detail::AdmitPick(pick);
  if (pick_token == nullptr || pick_token->ops == nullptr ||
      !pick_token->ops->resident) {
    return detail::RejectContext("accel_context_pick_invalid");
  }

  const rund::RuntimeStats stats = detail::ReadBackendStats(pick_token);
  rund::AccelDevice canonical_pick = pick_token->raw;
  canonical_pick.owner = detail::PublicPickOwner(pick_token);
  std::shared_ptr<detail::ContextToken> token = detail::MintContextToken(
      canonical_pick.api, canonical_pick.caps, pick_token);
  if (token == nullptr) {
    return detail::RejectContext("accel_context_pick_invalid");
  }
  return rund::AccelContext{
      .check = rund::AccelCheck{true, "ok"},
      .id = token->id,
      .pick = canonical_pick,
      .api = canonical_pick.api,
      .caps = canonical_pick.caps,
      .owner = detail::PublicTokenOwner(token),
      .evidence =
          rund::AccelContextEvidence{
              .api = canonical_pick.api,
              .caps = canonical_pick.caps,
              .runtime_stats = stats,
              .ok = true,
              .reason = "ok",
          },
  };
}

} // namespace rund::node::accel
