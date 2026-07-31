#pragma once

#include <kernel/core/model.hpp>

namespace rund::kernel {

using DispatchFn = void (*)(void* context, const Partition& partition);

struct WorkerTask {
  void* context = nullptr;
  DispatchFn invoke = nullptr;

  [[nodiscard]] explicit operator bool() const { return invoke != nullptr; }
};

} // namespace rund::kernel
