#pragma once

#include "capability.hpp"
#include "request.hpp"
#include "validation.hpp"

#include <memory>

namespace rund::node::accel::detail {

using SubmitServiceFreeDirect =
    rund::AccelCheck (*)(const ServiceFreeDirectRequest &) noexcept;

struct ServiceFreeDirectPreparation final {
  ServiceFreeDirectCapability capability{};
  std::shared_ptr<void> lowering{};
  SubmitServiceFreeDirect submit{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return service_free_direct_capable(capability) && lowering != nullptr &&
           submit != nullptr;
  }
};

} // namespace rund::node::accel::detail
