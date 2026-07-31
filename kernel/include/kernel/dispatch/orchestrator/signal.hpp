#pragma once

#include <atomic>

namespace rund::kernel {

struct FailureSignal {
  std::atomic<bool> failed{false};
  std::atomic<const char *> reason{"pass"};
};

} // namespace rund::kernel
