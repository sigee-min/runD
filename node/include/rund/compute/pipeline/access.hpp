#pragma once

#include <rund/compute/device.hpp>
#include <rund/compute/program.hpp>

#include <memory>

namespace rund::compute::detail {

struct DeviceAccess final {
  [[nodiscard]] static const std::shared_ptr<DeviceState> &
  state(const Device &device) noexcept {
    return device.state_;
  }
};

struct ProgramAccess final {
  template <class Signature>
  [[nodiscard]] static const std::shared_ptr<ProgramState> &
  state(const Program<Signature> &program) noexcept {
    return program.state_;
  }
};

} // namespace rund::compute::detail
