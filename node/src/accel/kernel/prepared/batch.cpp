#include "../prepared.hpp"

#include "evidence.hpp"
#include "model.hpp"

#include "../evidence.hpp"
#include "../reset/stats.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
#include <span>

namespace rund::node::accel::detail {
namespace {

class BatchUse final {
public:
  [[nodiscard]] bool
  claim(const std::span<prepared::RunState *const> states) noexcept {
    if (states.empty() || states.size() > submissions_.size()) {
      return false;
    }
    for (std::size_t index = 0u; index < states.size(); ++index) {
      prepared::RunSubmission &submission = states[index]->submission;
      locks_[index] =
          std::unique_lock<std::mutex>{submission.mutex, std::try_to_lock};
      if (!locks_[index].owns_lock() || submission.active) {
        return false;
      }
      submissions_[index] = &submission;
    }
    for (std::size_t index = 0u; index < states.size(); ++index) {
      submissions_[index]->active = true;
    }
    size_ = states.size();
    for (std::size_t index = 0u; index < size_; ++index) {
      locks_[index].unlock();
    }
    return true;
  }

  BatchUse(const BatchUse &) = delete;
  BatchUse &operator=(const BatchUse &) = delete;

  BatchUse() noexcept = default;
  ~BatchUse() { release(); }

private:
  void release() noexcept {
    for (std::size_t index = 0u; index < size_; ++index) {
      std::lock_guard lock{submissions_[index]->mutex};
      submissions_[index]->active = false;
    }
  }

  std::array<prepared::RunSubmission *, prepared::BatchCapacity> submissions_{};
  std::array<std::unique_lock<std::mutex>, prepared::BatchCapacity> locks_{};
  std::size_t size_{};
};

} // namespace

PreparedBatchEvidence
RunPreparedKernelBatch(const rund::AccelContext &context,
                       const std::span<const PreparedKernelRun *const> runs,
                       const std::span<rund::AccelEvidence> jobs,
                       std::shared_ptr<void> &workspace,
                       const PreparedBatchStart start, void *const user) {
  const rund::AccelCheck invalid{false, "accel_kernel_run_invalid"};
  if (runs.empty() || runs.size() > prepared::BatchCapacity ||
      runs.size() != jobs.size() || start == nullptr || user == nullptr) {
    return PreparedBatchEvidence{.check = invalid};
  }

  std::array<prepared::RunState *, prepared::BatchCapacity> states{};
  std::array<BackendBatchEntry, prepared::BatchCapacity> entries{};
  std::array<rund::AccelCheck, prepared::BatchCapacity> checks{};
  std::array<rund::RuntimeStats, prepared::BatchCapacity> job_stats{};
  const BackendOps *ops = nullptr;
  prepared::EvidenceCounts counts{};
  for (std::size_t index = 0u; index < runs.size(); ++index) {
    const PreparedKernelRun *const item = runs[index];
    auto *const state =
        item == nullptr ? nullptr
                        : static_cast<prepared::RunState *>(item->owner.get());
    const BackendOps *const candidate =
        state == nullptr ? nullptr : state->bound.run.ops;
    if (item == nullptr || !item->ok || state == nullptr ||
        IsPipelinePrivatePreparation(state->mode) || candidate == nullptr ||
        candidate->run_batch == nullptr ||
        (ops != nullptr && ops != candidate) ||
        !prepared::MatchesContext(context, *state)) {
      std::fill(jobs.begin(), jobs.end(),
                RejectKernelEvidence(context, {}, invalid.reason));
      return PreparedBatchEvidence{.check = invalid};
    }
    ops = candidate;
    states[index] = state;
    prepared::Accumulate(counts, *state);
    job_stats[index] = rund::RuntimeStats{.ok = true, .reason = "ok"};
    entries[index] = BackendBatchEntry{.run = &state->bound.run,
                                       .prepared = &state->backend,
                                       .stats = &job_stats[index]};
  }

  BatchUse use{};
  if (!use.claim(
          std::span<prepared::RunState *const>{states.data(), runs.size()})) {
    const rund::AccelCheck busy{false, "compute_batch_busy"};
    for (std::size_t index = 0u; index < runs.size(); ++index) {
      jobs[index] =
          RejectKernelEvidence(context, states[index]->execution, busy.reason);
    }
    return PreparedBatchEvidence{.check = busy};
  }

  start(user);

  rund::RuntimeStats stats{.ok = true, .reason = "ok"};
  const rund::AccelCheck ran = ops->run_batch(
      std::span<const BackendBatchEntry>{entries.data(), runs.size()},
      std::span<rund::AccelCheck>{checks.data(), runs.size()}, workspace,
      stats);
  const rund::AccelCheck batch =
      !stats.ok ? rund::AccelCheck{false, stats.reason} : ran;

  for (std::size_t index = 0u; index < runs.size(); ++index) {
    const prepared::RunState &state = *states[index];
    const rund::AccelCheck check =
        !stats.ok ? rund::AccelCheck{false, stats.reason} : checks[index];
    rund::RuntimeStats &local = job_stats[index];
    local.dispatch_count = check.ok ? state.dispatch.final_dispatch_count : 0u;
    if (!check.ok) {
      SetResetStats(local, false, 0u, 0u);
    }
    jobs[index] = EvidenceFromStats(
        context, state.execution, local, state.dispatch.original_dispatch_count,
        state.dispatch.final_dispatch_count, check.ok, check.reason,
        state.roundtrip.internal_bytes, state.roundtrip.external_bytes,
        check.failed_batches, check.first_failed_batch, check.first_status);
  }

  return PreparedBatchEvidence{
      .shared = prepared::BatchEvidence(context, stats, counts, batch),
      .check = batch,
  };
}

} // namespace rund::node::accel::detail
