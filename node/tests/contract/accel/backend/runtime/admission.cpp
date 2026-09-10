#include "local.hpp"

#include "src/accel/backend/token.hpp"
#include <node/accel/pick.hpp>

#include <memory>

namespace node_accel_contract::backend_runtime {
namespace {

namespace detail = rund::node::accel::detail;

} // namespace

bool CheckPickTokenAdmission() {
  if (detail::AdmitPick(rund::AccelDevice{}) != nullptr) {
    return false;
  }
  rund::AccelDevice pick = Pick(rund::AccelApi::Cpu);
  std::shared_ptr<detail::PickToken> token = detail::AdmitPick(pick);
  if (!pick.check.ok || token == nullptr || token->ops == nullptr ||
      token->raw.api != pick.api || token->raw.backend.context == nullptr) {
    return false;
  }
  const detail::BackendOps *const ops = token->ops;
  token.reset();
  const std::shared_ptr<detail::PickToken> retained = detail::AdmitPick(pick);
  if (retained == nullptr || retained->ops != ops ||
      retained->raw.backend.context == nullptr) {
    return false;
  }
  pick.api = rund::AccelApi::Fake;
  return detail::AdmitPick(pick) == nullptr;
}

} // namespace node_accel_contract::backend_runtime
