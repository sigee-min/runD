#include <node/runtime/backend.hpp>

#include <kernel/dispatch/kernel.hpp>

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace rund::node {
namespace {

thread_local const void *g_active_pool = nullptr;

struct BuiltInPoolBackend {
  explicit BuiltInPoolBackend(const std::uint32_t requested_width)
      : width(requested_width) {
    capacity_milli.assign(width, 1000u);
    workers.reserve(width > 0u ? width - 1u : 0u);
    for (std::uint32_t worker = 1u; worker < width; ++worker) {
      workers.emplace_back([this, worker] {
        ready.fetch_add(1u, std::memory_order_release);
        ready.notify_one();
        WorkerLoop(worker);
      });
    }
    const std::uint32_t expected = width > 0u ? width - 1u : 0u;
    std::uint32_t seen = ready.load(std::memory_order_acquire);
    while (seen < expected) {
      ready.wait(seen, std::memory_order_acquire);
      seen = ready.load(std::memory_order_acquire);
    }
  }

  BuiltInPoolBackend(const BuiltInPoolBackend &) = delete;
  BuiltInPoolBackend &operator=(const BuiltInPoolBackend &) = delete;

  ~BuiltInPoolBackend() {
    stop.store(true, std::memory_order_release);
    generation.fetch_add(1u, std::memory_order_release);
    generation.notify_all();
    for (std::thread &worker : workers) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  struct ActiveScope {
    explicit ActiveScope(const void *const pool) noexcept
        : previous(g_active_pool) {
      g_active_pool = pool;
    }

    ~ActiveScope() { g_active_pool = previous; }

    const void *previous = nullptr;
  };

  bool Nested() const noexcept { return g_active_pool == this; }

  bool Run(const rund::kernel::Partition *const input_partitions,
           const rund::kernel::u32 input_partition_count,
           const rund::kernel::WorkerTask input_task,
           rund::kernel::WorkerStats *const out_stats) {
    if (input_partitions == nullptr || !static_cast<bool>(input_task) ||
        width == 0u || Nested()) {
      return false;
    }
    std::lock_guard<std::mutex> lock(run_mutex);
    partitions = input_partitions;
    partition_count = input_partition_count;
    task = input_task;
    failed.store(false, std::memory_order_relaxed);
    collect_partitions_per_worker = out_stats != nullptr;
    if (collect_partitions_per_worker) {
      partitions_per_worker.assign(width, 0u);
    } else {
      partitions_per_worker.clear();
    }
    live.store(width, std::memory_order_relaxed);
    const std::uint64_t next = generation.load(std::memory_order_relaxed) + 1u;
    generation.store(next, std::memory_order_release);
    generation.notify_all();
    Work(0u);
    std::uint32_t seen = live.load(std::memory_order_acquire);
    while (seen != 0u) {
      live.wait(seen, std::memory_order_acquire);
      seen = live.load(std::memory_order_acquire);
    }
    PublishStats(out_stats);
    return !failed.load(std::memory_order_acquire);
  }

  void WorkerLoop(const std::uint32_t worker_index) {
    std::uint64_t seen = 0u;
    for (;;) {
      std::uint64_t current = generation.load(std::memory_order_acquire);
      while (current == seen && !stop.load(std::memory_order_acquire)) {
        generation.wait(seen, std::memory_order_acquire);
        current = generation.load(std::memory_order_acquire);
      }
      if (stop.load(std::memory_order_acquire)) {
        return;
      }
      seen = current;
      Work(worker_index);
    }
  }

  void Work(const std::uint32_t worker_index) noexcept {
    ActiveScope active(this);
    std::uint32_t count = 0u;
    for (std::uint32_t index = worker_index; index < partition_count;
         index += width) {
      try {
        task.invoke(task.context, partitions[index]);
        ++count;
      } catch (...) {
        failed.store(true, std::memory_order_release);
      }
    }
    if (collect_partitions_per_worker &&
        worker_index < partitions_per_worker.size()) {
      partitions_per_worker[worker_index] = count;
    }
    if (live.fetch_sub(1u, std::memory_order_acq_rel) == 1u) {
      live.notify_one();
    }
  }

  void PublishStats(rund::kernel::WorkerStats *const out_stats) const {
    if (out_stats == nullptr) {
      return;
    }
    out_stats->worker_count = width;
    out_stats->participating_workers = 0u;
    out_stats->total_partitions_executed = partition_count;
    out_stats->claim_fetch_count = 0u;
    out_stats->claim_fetch_count_measured = true;
    out_stats->static_tile_map_used = true;
    out_stats->global_claim_sync_elided = true;
    for (const std::uint32_t count : partitions_per_worker) {
      if (count != 0u) {
        ++out_stats->participating_workers;
      }
    }
    if (!out_stats->partitions_per_worker_sink.empty() &&
        out_stats->partitions_per_worker_sink.size() >=
            partitions_per_worker.size()) {
      for (std::size_t index = 0u; index < partitions_per_worker.size();
           ++index) {
        out_stats->partitions_per_worker_sink[index] =
            partitions_per_worker[index];
      }
      out_stats->partitions_per_worker.clear();
    } else {
      out_stats->partitions_per_worker = partitions_per_worker;
    }
  }

  std::uint32_t width = 1u;
  std::vector<std::uint32_t> capacity_milli{};
  std::vector<std::thread> workers{};
  std::mutex run_mutex{};
  std::atomic<bool> stop{false};
  std::atomic<bool> failed{false};
  std::atomic<std::uint64_t> generation{0u};
  std::atomic<std::uint32_t> live{0u};
  std::atomic<std::uint32_t> ready{0u};
  const rund::kernel::Partition *partitions = nullptr;
  rund::kernel::u32 partition_count = 0u;
  rund::kernel::WorkerTask task{};
  std::vector<std::uint32_t> partitions_per_worker{};
  bool collect_partitions_per_worker = false;
};

rund::kernel::u32 BuiltInPoolWorkerCount(void *const context) {
  const auto *const backend = static_cast<const BuiltInPoolBackend *>(context);
  return backend != nullptr ? backend->width : 0u;
}

rund::kernel::WorkerAffinityPolicy BuiltInPoolAffinity(void *) {
  return rund::kernel::WorkerAffinityPolicy::Static;
}

rund::kernel::WorkerBackendCapabilities
BuiltInPoolCapabilities(void *const context) {
  const auto *const backend = static_cast<const BuiltInPoolBackend *>(context);
  if (backend == nullptr) {
    return rund::kernel::WorkerBackendCapabilities{};
  }
  return rund::kernel::WorkerBackendCapabilities{
      .backend_width = backend->width,
      .width_matches_request = true,
      .supports_static_partitions = true,
      .supports_static_tile_map = true,
      .supports_claim_free_static_tiles = true,
      .supports_no_alloc_worker_stats = true,
      .supports_strict_fp_fold = true,
      .is_nested = false,
      .affinity_is_truth = true,
      .affinity_truth_level = rund::kernel::WorkerTruthLevel::Verified,
      .worker_capacity_milli = backend->capacity_milli.data(),
      .worker_capacity_count =
          static_cast<rund::kernel::u32>(backend->capacity_milli.size()),
      .worker_capacity_truth_level = rund::kernel::WorkerTruthLevel::Verified,
      .affinity_policy = rund::kernel::WorkerAffinityPolicy::Static,
  };
}

bool BuiltInPoolIsNested(void *const context) {
  const auto *const backend = static_cast<const BuiltInPoolBackend *>(context);
  return backend != nullptr && backend->Nested();
}

bool BuiltInPoolExecute(void *const context,
                        const rund::kernel::Partition *const partitions,
                        const rund::kernel::u32 partition_count,
                        const rund::kernel::WorkerTask task,
                        rund::kernel::WorkerStats *const out_stats) {
  auto *const backend = static_cast<BuiltInPoolBackend *>(context);
  return backend != nullptr &&
         backend->Run(partitions, partition_count, task, out_stats);
}

} // namespace

BackendSelection select_backend(const kernel::WorkerBackend backend,
                                const std::uint32_t workers) {
  if (workers == 0u) {
    return BackendSelection{.code = ReasonCode::BackendWidthRequired};
  }

  if (!static_cast<bool>(backend)) {
    return BackendSelection{.code = ReasonCode::BackendInvalid,
                            .requested_worker_width = workers};
  }
  const rund::kernel::WorkerBackendCapabilities capabilities =
      rund::kernel::InspectWorkerBackend(backend, workers);
  if (capabilities.backend_width == 0u) {
    return BackendSelection{.code = ReasonCode::BackendWidthRequired,
                            .requested_worker_width = workers};
  }
  if (!capabilities.width_matches_request) {
    return BackendSelection{.code = ReasonCode::BackendInvalid,
                            .requested_worker_width = workers};
  }
  return BackendSelection{.code = ReasonCode::Ok,
                          .requested_worker_width = workers,
                          .backend = backend};
}

BackendSelection select_backend(const std::uint32_t workers) {
  if (workers == 0u) {
    return BackendSelection{.code = ReasonCode::BackendWidthRequired};
  }
  auto owner = std::make_shared<BuiltInPoolBackend>(workers);
  rund::kernel::WorkerBackend backend{
      .context = owner.get(),
      .worker_count = BuiltInPoolWorkerCount,
      .affinity_policy = BuiltInPoolAffinity,
      .capabilities = BuiltInPoolCapabilities,
      .is_nested = BuiltInPoolIsNested,
      .execute_partitions = BuiltInPoolExecute,
  };
  BackendSelection selection = select_backend(backend, workers);
  selection.owner = std::move(owner);
  selection.verified_numa = true;
  return selection;
}

} // namespace rund::node
