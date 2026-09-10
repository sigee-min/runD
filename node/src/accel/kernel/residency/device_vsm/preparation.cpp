#include "projection.hpp"

#include "../../../backend/token.hpp"

namespace rund::node::accel::detail {

DeviceVsmPreparation
PrepareDeviceVsm(const rund::AccelDevice &pick,
                 const std::shared_ptr<const DeviceVsmProof> &proof) noexcept {
  const std::shared_ptr<PickToken> token = AdmitPick(pick);
  return token == nullptr || token->ops == nullptr ||
                 token->ops->prepare_device_vsm == nullptr
             ? DeviceVsmPreparation{}
             : token->ops->prepare_device_vsm(token->raw, proof);
}

} // namespace rund::node::accel::detail
