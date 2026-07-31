#pragma once

#include <kernel/core/model.hpp>

#include <atomic>

namespace rund::kernel {

struct WorkerCompletion final {
  void* context = nullptr;
  void (*invoke)(void* context, bool ok) noexcept = nullptr;
};

struct WorkerSubmission final {
  std::atomic<u32> remaining{0u};
  std::atomic_bool failed{false};
  WorkerCompletion completion{};
};

}  // namespace rund::kernel
