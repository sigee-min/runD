#include "execution.hpp"

#include "backing.hpp"
#include "backing_set.hpp"
#include "device_vsm/model.hpp"
#include "device_vsm/operations.hpp"
#include "device_vsm/route.hpp"
#include "dispatch.hpp"
#include "evidence.hpp"
#include "final.hpp"

#include "../../backend.hpp"
#include "../../pipeline/state.hpp"

#include <array>
#include <mutex>
#include <span>

namespace rund::compute::detail {
namespace {

struct VirtualAsyncFrame final {
  std::shared_ptr<VirtualPipelineState> state{};
  std::unique_lock<std::mutex> state_lock{};
  Stats stats{};
  VirtualRunTransaction transaction{};
  VirtualRunResources resources{};
  VirtualRunProjection projection{};
  VirtualRunWork work{};
  std::array<VirtualBacking *, VirtualPipelineState::InputCapacity> inputs{};
  VirtualBacking *output{};
  const VirtualVsmAsyncOps *vsm{};
  device_vsm_product_detail::DeviceVsmProductRun device{};
  Status status{Status::fail(Reason::CompletionInvalid)};
  std::uint64_t failed_page{ResidencyStats::no_failed_page};
  bool poison_pipeline{};
  bool started{};
};

[[nodiscard]] VirtualRunDispatchResult
finish_async_device(VirtualAsyncFrame &frame) noexcept {
  const bool open =
      frame.started || (frame.vsm != nullptr && frame.vsm->open != nullptr &&
                        frame.vsm->open(frame.device));
  VirtualExecutionResult executed =
      open
          ? (frame.vsm != nullptr && frame.vsm->finish != nullptr
                 ? frame.vsm->finish(
                       frame.device,
                       frame.started ? Status::success() : frame.status)
                 : VirtualExecutionResult{
                       .status = Status::fail(Reason::DeviceLost),
                       .failed_page = frame.device.failed_page,
                       .poison_pipeline = true,
                       .certainty = VirtualRunWriteCertainty::UnknownMayWrite,
                   })
          : VirtualExecutionResult{
                .status = frame.status,
                .failed_page = frame.device.failed_page,
                .poison_pipeline = frame.device.poison_pipeline,
                .certainty = VirtualRunWriteCertainty::KnownNoWrite,
            };
  VirtualRunDispatchResult result = dispatch_result(executed);
  result.poison_pipeline = result.poison_pipeline || frame.poison_pipeline ||
                           frame.device.poison_pipeline;
  return result;
}

} // namespace

VirtualAsyncStart
start_virtual_pipeline_async(const std::shared_ptr<VirtualPipelineState> &state,
                             void *const wake_user,
                             const VirtualWake wake) noexcept {
  VirtualAsyncStart result{};
  std::uint64_t active_count = 0u;
  if (state != nullptr && state->input_count != 0u &&
      state->input_count <= VirtualPipelineState::InputCapacity) {
    const VirtualBufferState *const input = virtual_input(*state);
    active_count = input == nullptr ? 0u : input->count;
  }
  std::unique_lock<std::mutex> state_lock;
  const VirtualRunAdmission admission =
      admit_virtual_run(state, active_count, state_lock,
                        VirtualRunAdmissionMode::AsynchronousDeviceVsm);
  if (!admission.claim) {
    result.status = admission.status;
    return result;
  }
  std::shared_ptr<VirtualAsyncFrame> frame;
  try {
    frame = std::make_shared<VirtualAsyncFrame>();
  } catch (...) {
    result.status = Status::fail(Reason::PipelineCapacity);
    return result;
  }
  frame->state = state;
  frame->output =
      state->output == nullptr ? nullptr : state->output->backing.get();
  frame->device.wake_user = wake_user;
  frame->device.wake = wake;
  frame->state_lock = std::move(state_lock);
  state->phase = VirtualPipelinePhase::Running;
  frame->stats = begin_virtual_run_evidence(*state, active_count);
  frame->failed_page = ResidencyStats::no_failed_page;
  frame->status = admission.status;
  frame->poison_pipeline = admission.poison_pipeline;
  if (!admission.status) {
    result.frame = frame;
    result.status = frame->status;
    return result;
  }
  if (!lock_backings(*state, frame->resources)) {
    frame->status = Status::fail(Reason::PipelineInvalid);
    frame->poison_pipeline = true;
    result.frame = frame;
    result.status = frame->status;
    return result;
  }
  if (!project_virtual_run(*state, active_count, frame->projection)) {
    frame->status = Status::fail(Reason::PipelineInvalid);
    frame->poison_pipeline = true;
    result.frame = frame;
    result.status = frame->status;
    return result;
  }
  frame->output = state->output->backing.get();
  for (std::size_t index = 0u; index < state->input_count; ++index) {
    frame->inputs[index] = virtual_input(*state, index)->backing.get();
  }
  const std::span<VirtualBacking *const> inputs{frame->inputs.data(),
                                                state->input_count};
  if (frame->output == nullptr ||
      !validate_virtual_run_recovery(frame->projection, inputs,
                                     *frame->output)) {
    frame->status = Status::fail(Reason::PipelineInvalid);
    result.frame = frame;
    result.status = frame->status;
    return result;
  }

  const VirtualDeviceVsmCandidate candidate = probe_virtual_device_vsm_route(
      *state, inputs, *frame->output, frame->projection);
  if (candidate.ordinary()) {
    state->phase = VirtualPipelinePhase::Ready;
    frame->state_lock.unlock();
    result.status = Status::fail(Reason::BackendUnsupported);
    return result;
  }
  if (candidate.terminal()) {
    frame->status = candidate.status;
    result.frame = frame;
    result.status = frame->status;
    return result;
  }

  const DeviceOps *const vsm_ops = state->pipeline->device->ops;
  if (vsm_ops == nullptr || vsm_ops->virtual_execution.vsm_async == nullptr ||
      !vsm_ops->virtual_execution.vsm_async->ready()) {
    frame->status = Status::fail(Reason::DeviceLost);
    frame->poison_pipeline = true;
    result.frame = frame;
    result.status = frame->status;
    return result;
  }
  frame->vsm = vsm_ops->virtual_execution.vsm_async;

  const Status pool_locked = frame->projection.poolless_device_vsm()
                                 ? Status::success()
                                 : lock_pool(*state, frame->resources);
  if (!pool_locked) {
    frame->status = pool_locked;
    result.frame = frame;
    result.status = frame->status;
    return result;
  }
  if (!frame->projection.poolless_device_vsm()) {
    const Status work = prepare_work(frame->projection, frame->work);
    if (!work) {
      frame->status = work;
      result.frame = frame;
      result.status = work;
      return result;
    }
  }

  VirtualDeviceVsmPostStage post =
      prepare_virtual_device_vsm_route(*state, frame->projection, candidate);
  const VirtualDeviceVsmScope scope = frame->projection.poolless_device_vsm()
                                          ? VirtualDeviceVsmScope::Poolless
                                          : VirtualDeviceVsmScope::Pooled;
  if (!post.ready()) {
    const VirtualRunDispatchResult failure =
        dispose_device_vsm_route(*frame->state, frame->resources, post, scope,
                                 post.status, frame->poison_pipeline);
    frame->status = failure.status;
    frame->poison_pipeline = failure.poison_pipeline;
    result.frame = frame;
    result.status = frame->status;
    return result;
  }
  if (!frame->projection.poolless_device_vsm()) {
    const Status pooled = stage_pool(*state, frame->projection, frame->stats,
                                     frame->resources, frame->poison_pipeline);
    if (!pooled) {
      const VirtualRunDispatchResult failure =
          dispose_device_vsm_route(*frame->state, frame->resources, post, scope,
                                   pooled, frame->poison_pipeline);
      frame->status = failure.status;
      frame->poison_pipeline = failure.poison_pipeline;
      result.frame = frame;
      result.status = frame->status;
      return result;
    }
  }
  if (scope == VirtualDeviceVsmScope::Pooled) {
    const Status closed = close_pool_stage(frame->resources);
    if (!closed) {
      const VirtualRunDispatchResult failure =
          dispose_device_vsm_route(*frame->state, frame->resources, post, scope,
                                   closed, frame->poison_pipeline);
      frame->status = failure.status;
      frame->poison_pipeline = failure.poison_pipeline;
      result.frame = frame;
      result.status = frame->status;
      return result;
    }
  }
  const Status rearmed = rearm_virtual_device_vsm_route(post);
  if (!rearmed) {
    const VirtualRunDispatchResult failure =
        dispose_device_vsm_route(*frame->state, frame->resources, post, scope,
                                 rearmed, frame->poison_pipeline);
    frame->status = failure.status;
    frame->poison_pipeline = failure.poison_pipeline;
    result.frame = frame;
    result.status = frame->status;
    return result;
  }
  if (!post.consume()) {
    const VirtualRunDispatchResult failure = dispose_device_vsm_route(
        *frame->state, frame->resources, post, scope,
        Status::fail(Reason::PipelineInvalid), frame->poison_pipeline);
    frame->status = failure.status;
    frame->poison_pipeline = failure.poison_pipeline;
    result.frame = frame;
    result.status = frame->status;
    return result;
  }
  frame->device.wait_for_callback = false;
  const Status started =
      frame->vsm->start(*state, inputs, *frame->output, frame->projection,
                        post.prepared, frame->stats, frame->device);
  frame->status = started;
  if (!started) {
    frame->poison_pipeline = frame->device.poison_pipeline;
    result.frame = frame;
    result.status = started;
    return result;
  }
  frame->started = true;
  result.frame = frame;
  result.status = Status::success();
  result.submitted = true;
  return result;
}

Status resume_virtual_pipeline_async(std::shared_ptr<void> &raw,
                                     Stats &stats) noexcept {
  auto frame = std::static_pointer_cast<VirtualAsyncFrame>(raw);
  if (frame == nullptr || frame->state == nullptr || frame->output == nullptr) {
    return Status::fail(Reason::CompletionInvalid);
  }
  if (frame->started &&
      (frame->vsm == nullptr || frame->vsm->valid == nullptr ||
       !frame->vsm->valid(frame->device))) {
    if (frame->vsm != nullptr && frame->vsm->unknown != nullptr) {
      frame->vsm->unknown(frame->device);
    } else {
      frame->device.poison_pipeline = true;
    }
  }
  const VirtualRunDispatchResult dispatched = finish_async_device(*frame);
  const Status result = finish_virtual_dispatch(
      *frame->state, std::move(frame->stats), frame->transaction,
      frame->projection, *frame->output, frame->work.reduction, dispatched,
      frame->failed_page, frame->poison_pipeline, &frame->resources);
  stats = frame->state_lock.owns_lock() ? frame->state->stats
                                        : virtual_pipeline_stats(frame->state);
  if (frame->state_lock.owns_lock()) {
    frame->state_lock.unlock();
  }
  raw.reset();
  return result;
}

} // namespace rund::compute::detail
