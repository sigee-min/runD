#include <kernel/program/executor/model.hpp>

#include <algorithm>
#include <limits>
#include <thread>

namespace rund::kernel {
namespace {

thread_local ParallelRuntimeProvider active_provider{};

} // namespace

namespace executor_detail {

ScopedParallelRuntimeProvider::ScopedParallelRuntimeProvider(
    const ParallelRuntimeProvider provider) noexcept
    : previous_(active_provider) {
  if (!provider) {
    reason_ = "parallel_runtime_provider_invalid";
    return;
  }
  active_provider = provider;
  installed_ = true;
  reason_ = "ok";
}

ScopedParallelRuntimeProvider::~ScopedParallelRuntimeProvider() {
  if (installed_) {
    active_provider = previous_;
  }
}

u32 HostParallelWorkerHint() noexcept {
  const unsigned int hardware = std::thread::hardware_concurrency();
  if (hardware == 0u) {
    return 1u;
  }
  return static_cast<u32>(std::min<unsigned int>(
      hardware,
      static_cast<unsigned int>(std::numeric_limits<u32>::max())));
}

bool ParallelRuntimeProviderActive() noexcept {
  return static_cast<bool>(active_provider);
}

ParallelRuntime AcquireParallelRuntime(const u32 workers) noexcept {
  const ParallelRuntimeProvider provider = active_provider;
  if (!provider) {
    return ParallelRuntime{
        .workers = workers,
        .reason = "parallel_runtime_missing",
    };
  }
  const ParallelRuntime runtime = provider.acquire(provider.context, workers);
  if (!runtime.valid) {
    return runtime;
  }
  if (runtime.workspace == nullptr) {
    return ParallelRuntime{
        .workers = workers,
        .reason = "parallel_runtime_workspace_missing",
    };
  }
  if (!runtime.worker_backend) {
    return ParallelRuntime{
        .workspace = runtime.workspace,
        .workers = workers,
        .reason = "parallel_runtime_backend_invalid",
    };
  }
  return runtime;
}

} // namespace executor_detail
} // namespace rund::kernel
