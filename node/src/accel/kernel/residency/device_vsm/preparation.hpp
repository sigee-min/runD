#pragma once

#include "capability.hpp"
#include "request.hpp"
#include "validation.hpp"

#include <memory>

namespace rund::node::accel::detail {

using SubmitDeviceVsm = rund::AccelCheck (*)(const DeviceVsmRequest &) noexcept;
struct DeviceVsmRearmResult final {
  rund::AccelCheck check{};
  bool mutated{};
};

using RearmDeviceVsm = DeviceVsmRearmResult (*)(
    const std::shared_ptr<void> &,
    const std::shared_ptr<const DeviceVsmProof> &) noexcept;

struct DeviceVsmPreparation final {
  DeviceVsmCapability capability{};
  std::shared_ptr<void> lowering{};
  SubmitDeviceVsm submit{};
  RearmDeviceVsm rearm{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return device_vsm_capable(capability) && lowering != nullptr &&
           submit != nullptr && rearm != nullptr;
  }
};

} // namespace rund::node::accel::detail
