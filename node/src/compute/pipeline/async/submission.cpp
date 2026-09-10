#include "../../../accel/kernel/prepared/interface/evidence.hpp"
#include "../local.hpp"

#include <rund/compute/pipeline.hpp>
#include <rund/counter.hpp>

#include "../../backend.hpp"
#include "../../device/info.hpp"
#include "../../job/local.hpp"
#include "../../stats.hpp"
#include "../../status.hpp"
#include "../../terminal.hpp"
#include "../claim.hpp"
#include "../run/memory.hpp"
#include "../run/result.hpp"

#include <algorithm>
#include <mutex>
#include <utility>

namespace rund::compute::detail {

Result<Backend>
pipeline_backend(const std::shared_ptr<PipelineState> &state) noexcept {
  return !valid_pipeline(state)
             ? Result<Backend>::fail(Reason::PipelineInvalid)
             : Result<Backend>::success(state->device->backend);
}

kernel::u32
pipeline_workers(const std::shared_ptr<PipelineState> &state) noexcept {
  if (!valid_pipeline(state)) {
    return 0u;
  }
  const CpuDeviceState *const cpu = cpu_device(*state->device);
  return cpu == nullptr ? 0u : cpu->workers.requested_worker_width;
}

Status queue_pipeline(const std::shared_ptr<PipelineState> &state) noexcept {
  if (!valid_pipeline(state)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard lock{state->gate};
  return start_pipeline(*state);
}

std::size_t
pipeline_size(const std::shared_ptr<PipelineState> &state) noexcept {
  return state == nullptr ? 0u : state->steps.size();
}

std::shared_ptr<JobState>
pipeline_job(const std::shared_ptr<PipelineState> &state,
             const std::size_t index) noexcept {
  return state == nullptr || index >= state->steps.size()
             ? std::shared_ptr<JobState>{}
         : state->transactional && state->attempt.parity != 0u
             ? state->steps[index].alternate_job
             : state->steps[index].job;
}

Status submit_pipeline_on(
    const std::shared_ptr<PipelineState> &state, std::shared_ptr<void> lifetime,
    const PipelineCompletion completion, void *const user,
    const node::accel::detail::KernelTiming timing,
    const node::accel::detail::PipelineSubmitMode mode) noexcept {
  if (!valid_pipeline(state) || completion == nullptr || user == nullptr ||
      state->device->backend == Backend::Cpu) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const DeviceOps *const ops = state->device->ops;
  if (ops == nullptr || ops->submit_pipeline == nullptr) {
    return Status::fail(Reason::AccelProgramInvalid);
  }
  const node::accel::detail::PreparedKernelPipeline *prepared = nullptr;
  {
    std::unique_lock lock{state->gate};
    if (state->phase != PipelinePhase::Running) {
      return Status::fail(Reason::PipelineBusy);
    }
    if (state->active_step_count == 0u) {
      node::accel::detail::PreparedPipelineEvidence empty{};
      empty.check = {true, "ok"};
      empty.shared.identity.backend = state->device->backend == Backend::Metal
                                          ? rund::AccelApi::Metal
                                          : rund::AccelApi::Vulkan;
      empty.shared.outcome.ok = true;
      empty.shared.outcome.reason = "ok";
      lock.unlock();
      completion(user, std::move(empty));
      return Status::success();
    }
    prepared = state->transactional && state->attempt.parity != 0u
                   ? &state->alternate_prepared
                   : &state->prepared;
    if (!prepared->ok) {
      return Status::fail(Reason::PipelineInvalid);
    }
    state->attempt.dispatch_timing =
        timing == node::accel::detail::KernelTiming::Dispatch;
  }
  const rund::AccelCheck submitted =
      ops->submit_pipeline(*state->device, *prepared, std::move(lifetime),
                           completion, user, timing, mode);
  std::lock_guard lock{state->gate};
  if (!submitted.ok) {
    return Status::fail(
        project_reason(submitted.reason, Reason::BackendFailed));
  }
  state->attempt.backend_submitted = true;
  return Status::success();
}

void record_pipeline_frame(const std::shared_ptr<PipelineState> &state,
                           const std::uint64_t bytes, const bool reused,
                           const std::uint64_t budget) noexcept {
  if (state == nullptr) {
    return;
  }
  std::lock_guard lock{state->gate};
  ::rund::detail::counter::Accumulate(state->frame_current, bytes);
  state->frame_peak = std::max(state->frame_peak, state->frame_current);
  ::rund::detail::counter::Accumulate(state->frame_bytes, bytes);
  ::rund::detail::counter::Accumulate(state->frame_reused, reused ? bytes : 0u);
  state->frame_budget = std::max(state->frame_budget, budget);
}

} // namespace rund::compute::detail
