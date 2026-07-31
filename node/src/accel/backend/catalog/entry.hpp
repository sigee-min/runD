#pragma once

#include <accel/api.hpp>

namespace rund {
struct AccelDevice;
}

namespace rund::node::accel::detail {

struct BackendOps;

struct BackendEntry final {
  rund::AccelApi api = rund::AccelApi::Auto;
  bool automatic = false;
  rund::AccelDevice (*pick)(bool allow_fake) = nullptr;
  const BackendOps *ops = nullptr;
};

} // namespace rund::node::accel::detail
