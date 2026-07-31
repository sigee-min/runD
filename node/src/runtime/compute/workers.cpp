#include "workers.hpp"

#include "../compute/state.hpp"

#include <atomic>
#include <cstdint>

namespace rund::node::runtime_detail {
namespace {

kernel::u32 WorkerCount(void* const raw) noexcept {
  const auto* const host = static_cast<const ComputeHostState*>(raw);
  return host == nullptr ? 0u : host->workers;
}

kernel::WorkerAffinityPolicy Affinity(void*) noexcept {
  return kernel::WorkerAffinityPolicy::Static;
}

kernel::WorkerBackendCapabilities Capabilities(void* const raw) noexcept {
  const auto* const host = static_cast<const ComputeHostState*>(raw);
  if (host == nullptr) {
    return {};
  }
  return kernel::WorkerBackendCapabilities{
      .backend_width = host->workers,
      .width_matches_request = true,
      .supports_static_partitions = true,
      .supports_async_partitions = true,
      .supports_static_tile_map = true,
      .supports_claim_free_static_tiles = true,
      .supports_no_alloc_worker_stats = true,
      .supports_strict_fp_fold = true,
      .is_nested = false,
      .affinity_is_truth = true,
      .affinity_truth_level = kernel::WorkerTruthLevel::Verified,
      .worker_capacity_milli = host->worker_capacity.data(),
      .worker_capacity_count =
          static_cast<kernel::u32>(host->worker_capacity.size()),
      .worker_capacity_truth_level = kernel::WorkerTruthLevel::Verified,
      .affinity_policy = kernel::WorkerAffinityPolicy::Static,
  };
}

bool IsNested(void*) noexcept { return false; }

bool SyncNested(void* const raw) noexcept {
  auto* const host = static_cast<ComputeHostState*>(raw);
  return host != nullptr && Scheduler::Active() == &host->scheduler &&
         host->scheduler.ActiveState().task_id != 0u;
}

void PublishStats(const ComputeWorkerBatch& batch) noexcept {
  kernel::WorkerStats* const stats = batch.stats;
  if (stats == nullptr) {
    return;
  }
  stats->worker_count = batch.workers;
  stats->participating_workers = 0u;
  stats->total_partitions_executed = 0u;
  stats->claim_fetch_count = 0u;
  stats->claim_fetch_count_measured = true;
  stats->static_tile_map_used = true;
  stats->global_claim_sync_elided = true;
  for (kernel::u32 worker = 0u; worker < batch.workers; ++worker) {
    const kernel::u32 count = batch.slots[worker].partitions;
    stats->total_partitions_executed += count;
    stats->participating_workers += count == 0u ? 0u : 1u;
    if (worker < stats->partitions_per_worker_sink.size()) {
      stats->partitions_per_worker_sink[worker] = count;
    }
  }
  stats->partitions_per_worker.clear();
}

void FinishWorker(ComputeWorkerBatch& batch, const bool failed) noexcept {
  kernel::WorkerSubmission& submission = *batch.submission;
  if (failed) {
    submission.failed.store(true, std::memory_order_release);
  }
  if (submission.remaining.fetch_sub(1u, std::memory_order_acq_rel) != 1u) {
    return;
  }
  PublishStats(batch);
  if (submission.completion.invoke != nullptr) {
    const kernel::WorkerCompletion completion = submission.completion;
    const bool ok = !submission.failed.load(std::memory_order_acquire);
    batch.partitions = nullptr;
    batch.partition_count = 0u;
    batch.task = {};
    batch.submission = nullptr;
    batch.stats = nullptr;
    if (batch.host != nullptr) {
      (void)batch.host->worker_batch_slots.release(batch.slot);
    }
    completion.invoke(completion.context, ok);
  }
}

void RunSlot(void* const raw) noexcept {
  auto* const slot = static_cast<ComputeWorkerSlot*>(raw);
  ComputeWorkerBatch* const batch = slot == nullptr ? nullptr : slot->batch;
  if (batch == nullptr || batch->submission == nullptr) {
    return;
  }
  bool failed = false;
  kernel::u32 completed = 0u;
  for (kernel::u32 index = slot->worker; index < batch->partition_count;
       index += batch->workers) {
    try {
      batch->task.invoke(batch->task.context, batch->partitions[index]);
      ++completed;
    } catch (...) {
      failed = true;
    }
  }
  slot->partitions = completed;
  slot->failed = failed;
}

void ReleaseSlot(void* const raw) noexcept {
  auto* const slot = static_cast<ComputeWorkerSlot*>(raw);
  if (slot == nullptr || slot->batch == nullptr) {
    return;
  }
  FinishWorker(*slot->batch, slot->failed);
}

ComputeWorkerBatch* ClaimBatch(ComputeHostState& host) noexcept {
  std::lock_guard lock{host.mutex};
  const std::optional<std::size_t> claimed =
      host.worker_batch_slots.claim(host.worker_batches.size());
  if (!claimed.has_value()) {
    return nullptr;
  }
  if (*claimed < host.worker_batches.size()) {
    return host.worker_batches[*claimed].get();
  }
  try {
    auto batch = std::make_unique<ComputeWorkerBatch>();
    batch->host = &host;
    batch->slot = *claimed;
    batch->slots.resize(host.workers);
    for (kernel::u32 worker = 0u; worker < host.workers; ++worker) {
      batch->slots[worker].batch = batch.get();
      batch->slots[worker].worker = worker;
    }
    ComputeWorkerBatch *const claimed = batch.get();
    host.worker_batches.push_back(std::move(batch));
    return claimed;
  } catch (...) {
    return nullptr;
  }
}

bool Submit(void* const raw,
            const kernel::Partition* const partitions,
            const kernel::u32 partition_count,
            const kernel::WorkerTask task,
            kernel::WorkerStats* const stats,
            kernel::WorkerSubmission* const submission) noexcept {
  auto* const host = static_cast<ComputeHostState*>(raw);
  if (host == nullptr || partitions == nullptr || !task ||
      submission == nullptr || host->workers == 0u ||
      submission->completion.invoke == nullptr) {
    return false;
  }
  ComputeWorkerBatch* const batch = ClaimBatch(*host);
  if (batch == nullptr || batch->slots.size() < host->workers) {
    if (batch != nullptr) {
      (void)host->worker_batch_slots.release(batch->slot);
    }
    return false;
  }
  const kernel::u32 workers = host->workers;
  batch->partitions = partitions;
  batch->partition_count = partition_count;
  batch->workers = workers;
  batch->task = task;
  batch->submission = submission;
  batch->stats = stats;
  submission->failed.store(false, std::memory_order_relaxed);
  submission->remaining.store(workers + 1u, std::memory_order_release);
  for (kernel::u32 worker = 0u; worker < workers; ++worker) {
    ComputeWorkerSlot& slot = batch->slots[worker];
    slot.partitions = 0u;
    slot.failed = false;
    slot.work = SchedulerWork{
        .context = &slot,
        .invoke = RunSlot,
        .released = ReleaseSlot,
    };
    if (!host->scheduler.EnqueueWork(&slot.work, worker)) {
      FinishWorker(*batch, true);
    }
  }
  FinishWorker(*batch, false);
  return true;
}

struct SyncWait final {
  std::atomic_bool done{false};
  bool ok = false;
};

void FinishSync(void* const raw, const bool ok) noexcept {
  auto* const wait = static_cast<SyncWait*>(raw);
  if (wait == nullptr) {
    return;
  }
  wait->ok = ok;
  wait->done.store(true, std::memory_order_release);
  wait->done.notify_one();
}

bool Execute(void* const raw, const kernel::Partition* const partitions,
             const kernel::u32 partition_count, const kernel::WorkerTask task,
             kernel::WorkerStats* const stats) noexcept {
  auto* const host = static_cast<ComputeHostState*>(raw);
  if (host == nullptr || SyncNested(host)) {
    return false;
  }
  SyncWait wait{};
  kernel::WorkerSubmission submission{};
  submission.completion = kernel::WorkerCompletion{
      .context = &wait,
      .invoke = FinishSync,
  };
  if (!Submit(host, partitions, partition_count, task, stats, &submission)) {
    return false;
  }
  wait.done.wait(false, std::memory_order_acquire);
  return wait.ok;
}

}  // namespace

kernel::WorkerBackend MakeComputeWorkerBackend(ComputeHostState& host) noexcept {
  return kernel::WorkerBackend{
      .context = &host,
      .worker_count = WorkerCount,
      .affinity_policy = Affinity,
      .capabilities = Capabilities,
      .is_nested = IsNested,
      .submit_partitions = Submit,
  };
}

kernel::WorkerBackend
MakeComputeSyncWorkerBackend(ComputeHostState& host) noexcept {
  return kernel::WorkerBackend{
      .context = &host,
      .worker_count = WorkerCount,
      .affinity_policy = Affinity,
      .capabilities = Capabilities,
      .is_nested = SyncNested,
      .execute_partitions = Execute,
  };
}

}  // namespace rund::node::runtime_detail
