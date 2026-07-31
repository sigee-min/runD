#include <rund/compute/batch.hpp>

#include "../../accel/kernel/prepared.hpp"
#include "../device/state.hpp"
#include "../job/local.hpp"
#include "../job/state.hpp"
#include "../stats.hpp"
#include "../status.hpp"

#include <accel/kernel/evidence.hpp>

#include <array>
#include <memory>
#include <mutex>
#include <span>

namespace rund::compute::detail {
namespace {

struct BatchStart final {
  std::span<const std::shared_ptr<JobState>> jobs{};
  std::span<std::unique_lock<std::mutex>> locks{};
  bool started = false;
};

[[nodiscard]] Status AdmitJob(
    const std::shared_ptr<JobState> &job) noexcept {
  if (!valid_job(job) || job->program == nullptr ||
      job->program->device == nullptr || job->program->empty()) {
    return Status::fail(Reason::BatchPreparedInvalid);
  }
  if (job->program->device->backend == Backend::Cpu) {
    return Status::fail(Reason::BatchCpuUnsupported);
  }
  return job->prepared.ok ? Status::success()
                          : Status::fail(Reason::BatchPreparedInvalid);
}

[[nodiscard]] Status ClaimJobs(
    const std::span<const std::shared_ptr<JobState>> jobs,
    std::array<std::unique_lock<std::mutex>, BatchCapacity> &locks) {
  for (std::size_t index = 0u; index < jobs.size(); ++index) {
    const std::shared_ptr<JobState> &job = jobs[index];
    locks[index] =
        std::unique_lock<std::mutex>{job->gate, std::try_to_lock};
    if (!locks[index].owns_lock() || job_busy(job->phase)) {
      return Status::fail(Reason::BatchBusy);
    }
  }
  return Status::success();
}

void StartJobs(void *const raw) noexcept {
  auto &start = *static_cast<BatchStart *>(raw);
  for (const std::shared_ptr<JobState> &job : start.jobs) {
    if (job->terminal != nullptr) {
      job->terminal->last.reset();
      job->terminal->failed_stats.reset();
    }
    job->phase = JobPhase::Running;
  }
  for (std::unique_lock<std::mutex> &lock : start.locks) {
    lock.unlock();
  }
  start.started = true;
}

[[nodiscard]] Status FinishBatch(
    const std::span<const std::shared_ptr<JobState>> jobs,
    const std::span<const rund::AccelEvidence> evidence,
    const node::accel::detail::PreparedBatchEvidence &batch,
    Stats &stats) {
  stats = stats_from_evidence(jobs.front()->program->device->backend,
                              batch.shared, 0u);
  Status result = batch.check.ok
                      ? Status::success()
                      : Status::fail(project_reason(batch.check.reason,
                                                    Reason::BackendFailed));
  for (std::size_t index = 0u; index < jobs.size(); ++index) {
    const Status finished =
        finish_job(jobs[index], finish_job_accel(jobs[index], evidence[index]));
    if (result && !finished) {
      result = finished;
    }
  }
  return result;
}

} // namespace

Status add_batch(const std::span<std::shared_ptr<JobState>> jobs,
                 std::size_t &size, const std::shared_ptr<JobState> &job,
                 Stats &stats, std::shared_ptr<void> &workspace) noexcept {
  const Status valid = AdmitJob(job);
  if (!valid) {
    return valid;
  }
  for (std::size_t index = 0u; index < size; ++index) {
    if (jobs[index] == job) {
      return Status::fail(Reason::BatchDuplicate);
    }
  }
  if (size == jobs.size()) {
    return Status::fail(Reason::BatchCapacity);
  }
  if (size != 0u &&
      jobs.front()->program->device != job->program->device) {
    return Status::fail(Reason::BatchDeviceMismatch);
  }
  jobs[size++] = job;
  stats = Stats{.backend = job->program->device->backend};
  workspace.reset();
  return Status::success();
}

Status run_batch(const std::span<const std::shared_ptr<JobState>> jobs,
                 Stats &stats, std::shared_ptr<void> &workspace) {
  if (jobs.empty()) {
    return Status::fail(Reason::BatchEmpty);
  }
  if (jobs.size() > BatchCapacity) {
    return Status::fail(Reason::BatchCapacity);
  }

  std::array<std::unique_lock<std::mutex>, BatchCapacity> locks{};
  const Status claimed = ClaimJobs(jobs, locks);
  if (!claimed) {
    return claimed;
  }

  const AccelDeviceState *const device =
      accel_device(*jobs.front()->program->device);
  if (device == nullptr) {
    const Status failure = Status::fail(Reason::BatchPreparedInvalid);
    for (const std::shared_ptr<JobState> &job : jobs) {
      (void)fail_job(job, failure);
    }
    return failure;
  }

  std::array<const node::accel::detail::PreparedKernelRun *, BatchCapacity>
      prepared{};
  std::array<rund::AccelEvidence, BatchCapacity> evidence{};
  for (std::size_t index = 0u; index < jobs.size(); ++index) {
    prepared[index] = &jobs[index]->prepared;
  }
  BatchStart start{
      .jobs = jobs,
      .locks = std::span<std::unique_lock<std::mutex>>{locks.data(),
                                                       jobs.size()},
  };
  const node::accel::detail::PreparedBatchEvidence batch =
      node::accel::detail::RunPreparedKernelBatch(
          device->context,
          std::span<const node::accel::detail::PreparedKernelRun *const>{
              prepared.data(), jobs.size()},
          std::span<rund::AccelEvidence>{evidence.data(), jobs.size()},
          workspace, StartJobs, &start);
  if (!start.started) {
    return Status::fail(
        project_reason(batch.check.reason, Reason::BackendFailed));
  }
  return FinishBatch(
      jobs, std::span<const rund::AccelEvidence>{evidence.data(), jobs.size()},
      batch, stats);
}

} // namespace rund::compute::detail
