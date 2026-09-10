#pragma once

#include "../coordinate.hpp"
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

struct PreparedKernelRun {
  std::shared_ptr<void> owner{};
  bool ok = false;
  const char *reason = "accel_kernel_run_invalid";
  std::uint32_t failed_node = NoNode;
};


} // namespace rund::node::accel::detail
