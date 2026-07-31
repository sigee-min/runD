#pragma once

#include <kernel/dispatch/orchestrator.hpp>
#include <kernel/program/skeleton.hpp>
#include <kernel/program/tile.hpp>
#include <kernel/schedule/workspace.hpp>

namespace rund::kernel {

template <std::size_t Rank>
struct PreparedEach;

struct SerialPolicy {
  bool valid = true;
  const char* reason = "pass";
};

struct ParallelPolicy {
  u32 workers = 1u;
  bool use_runtime_default_workers = false;
  Alignment boundary_alignment{};
  KernelProgramPhysicalTilePolicy physical_tile_policy{};
  bool valid = true;
  const char* reason = "pass";
};

struct ParallelRuntime {
  Workspace* workspace = nullptr;
  WorkerBackend worker_backend{};
  u32 workers = 1u;
  bool valid = false;
  const char* reason = "parallel_runtime_missing";
};

struct ParallelRuntimeProvider {
  void* context = nullptr;
  ParallelRuntime (*acquire)(void* context, u32 workers) = nullptr;

  [[nodiscard]] explicit operator bool() const noexcept {
    return context != nullptr && acquire != nullptr;
  }
};

struct Executor {
  Workspace* workspace = nullptr;
  WorkerBackend worker_backend{};
  u32 workers = 1u;
  Alignment boundary_alignment{};
  KernelProgramPhysicalTilePolicy physical_tile_policy{};
  bool collect_worker_stats = false;
  bool valid = false;
  const char* reason = "executor_not_validated";

  template <std::size_t Rank>
  [[nodiscard]] PreparedEach<Rank> prepare(const Space<Rank>& index_space) const;
};

namespace executor_detail {
class ScopedParallelRuntimeProvider {
 public:
  explicit ScopedParallelRuntimeProvider(ParallelRuntimeProvider provider) noexcept;
  ScopedParallelRuntimeProvider(const ScopedParallelRuntimeProvider&) = delete;
  ScopedParallelRuntimeProvider& operator=(const ScopedParallelRuntimeProvider&) = delete;
  ScopedParallelRuntimeProvider(ScopedParallelRuntimeProvider&&) = delete;
  ScopedParallelRuntimeProvider& operator=(ScopedParallelRuntimeProvider&&) = delete;
  ~ScopedParallelRuntimeProvider();

  [[nodiscard]] explicit operator bool() const noexcept { return installed_; }
  [[nodiscard]] const char* reason() const noexcept { return reason_; }

 private:
  ParallelRuntimeProvider previous_{};
  bool installed_ = false;
  const char* reason_ = "parallel_runtime_provider_invalid";
};

[[nodiscard]] u32 HostParallelWorkerHint() noexcept;
[[nodiscard]] bool ParallelRuntimeProviderActive() noexcept;
[[nodiscard]] ParallelRuntime AcquireParallelRuntime(u32 workers) noexcept;

} // namespace executor_detail

} // namespace rund::kernel
