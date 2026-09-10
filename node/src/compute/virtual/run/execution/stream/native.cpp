#include "internal.hpp"

#include "../../../stats.hpp"

#include "../../../../pipeline/local.hpp"
#include "../../../../pipeline/run/clock.hpp"
#include "../../../../device/residency/pool.hpp"

#include <thread>
#include <utility>

namespace rund::compute::detail::virtual_stream_detail {

[[nodiscard]] Status signal_stream_native(StreamWait &wait,
                                          const std::uint64_t epoch,
                                          const Status admission) noexcept {
  return wait.schedule != nullptr ? signal_pipeline_execution_schedule(
                                        *wait.schedule, epoch, admission)
         : wait.native != nullptr
             ? signal_pipeline_execution_window(*wait.native, epoch, admission)
             : Status::fail(Reason::PipelineInvalid);
}

[[nodiscard]] Status abort_stream_native(StreamWait &wait,
                                         const Status failure) noexcept {
  return wait.schedule != nullptr
             ? abort_pipeline_execution_schedule(*wait.schedule, failure)
         : wait.native != nullptr
             ? abort_pipeline_execution_window(*wait.native, failure)
             : Status::fail(Reason::PipelineInvalid);
}

void complete_stream_native(void *const raw,
                            PipelineWindowRelease &&pipeline) noexcept {
  auto *const wait = static_cast<StreamWait *>(raw);
  if (wait == nullptr || wait->stream == nullptr ||
      (wait->native == nullptr && wait->schedule == nullptr)) {
    return;
  }
  Status status = pipeline.release.status;
  std::uint64_t next = 0u;
  {
    std::lock_guard lock{wait->coordinator};
    wait->callback_on_caller =
        wait->callback_on_caller || std::this_thread::get_id() == wait->caller;
    const std::uint64_t epoch = pipeline.release.epoch;
    if (!wait->stream->release(pipeline.release)) {
      status = Status::fail(Reason::CompletionInvalid);
      wait->quarantine = true;
    }
    if (!wait->disposition && status) {
      status = wait->disposition;
    }
    if (pipeline.terminal_published && pipeline.reseeded &&
        wait->stats != nullptr) {
      const std::size_t bank = epoch % residency::execution::BankCapacity;
      const Status accumulated = accumulate_virtual_epoch(
          *wait->stats, pipeline_stats(wait->pipelines[bank]));
      if (!accumulated) {
        status = accumulated;
      }
    }
    if (status) {
      status = service_stream_output(*wait, epoch);
    }
    next = epoch + residency::execution::BankCapacity;
    const std::uint64_t limit = wait->schedule != nullptr
                                    ? wait->stream->lease().epochs
                                    : wait->chunk_first + wait->chunk_count;
    if (status && next < limit) {
      status = service_stream_input(*wait, next);
    }
    if (!status && wait->disposition) {
      wait->disposition = status;
      wait->failed_page = wait->failed_page == ResidencyStats::no_failed_page
                              ? epoch * wait->run->frame_capacity
                              : wait->failed_page;
      wait->quarantine =
          wait->quarantine ||
          pipeline.release.terminal ==
              residency::execution::TerminalKind::UnknownMayWrite;
    }
  }
  const std::uint64_t limit = wait->schedule != nullptr
                                  ? wait->stream->lease().epochs
                                  : wait->chunk_first + wait->chunk_count;
  if (next < limit) {
    const Status signalled = signal_stream_native(*wait, next, status);
    if (!signalled) {
      const Status aborted = abort_stream_native(*wait, signalled);
      std::lock_guard lock{wait->coordinator};
      if (wait->disposition) {
        wait->disposition = aborted ? signalled : aborted;
      }
      wait->quarantine = true;
    }
  }
}

void complete_stream_schedule_final(
    void *const raw,
    residency::execution::ScheduleEvidence &&evidence) noexcept {
  auto *const wait = static_cast<StreamWait *>(raw);
  if (wait == nullptr || wait->stream == nullptr) {
    return;
  }
  residency::execution::ScheduleEvidence final_evidence = std::move(evidence);
  residency::execution::StreamEvidence final{};
  Status failure = Status::success();
  bool poison = false;
  {
    std::lock_guard lock{wait->coordinator};
    if (wait->terminal) {
      return;
    }
    if (final_evidence.status && !wait->disposition &&
        final_evidence.terminal !=
            residency::execution::TerminalKind::UnknownMayWrite) {
      final_evidence.status = wait->disposition;
      final_evidence.terminal = residency::execution::TerminalKind::Known;
    }
    if (!wait->stream->schedule(final_evidence) ||
        !wait->stream->final(final)) {
      wait->disposition = Status::fail(Reason::CompletionInvalid);
      wait->quarantine = true;
    }
    wait->terminal = true;
    failure = wait->disposition;
    poison = wait->quarantine ||
             final_evidence.terminal ==
                 residency::execution::TerminalKind::UnknownMayWrite;
  }
  {
    std::lock_guard lock{wait->gate};
    wait->final = std::move(final);
    wait->failure = failure;
    wait->poison = poison;
    wait->completed = true;
    wait->ready.notify_one();
  }
}

void complete_stream_final(
    void *const raw, residency::execution::WindowEvidence &&evidence) noexcept {
  auto *const wait = static_cast<StreamWait *>(raw);
  if (wait == nullptr || wait->stream == nullptr) {
    return;
  }
  bool notify = false;
  residency::execution::StreamEvidence final{};
  Status failure = Status::success();
  bool poison = false;
  bool schedule = false;
  {
    std::lock_guard lock{wait->coordinator};
    if (wait->terminal) {
      return;
    }
    if (!wait->stream->chunk(evidence)) {
      const bool abandoned =
          evidence.terminal == residency::execution::TerminalKind::Known &&
          wait->stream->abandon_final(evidence);
      wait->disposition = Status::fail(Reason::CompletionInvalid);
      wait->quarantine = !abandoned;
      wait->terminal = true;
      notify = true;
    } else {
      if (wait->stream->final(final)) {
        wait->terminal = true;
        notify = true;
      } else {
        wait->pending = true;
        if (!wait->pumping && !wait->scheduled) {
          wait->scheduled = true;
          schedule = true;
        }
      }
    }
    if (notify) {
      failure = wait->disposition;
      poison = wait->quarantine;
    }
  }
  if (notify) {
    {
      std::lock_guard lock{wait->gate};
      wait->final = std::move(final);
      wait->failure = failure;
      wait->poison = poison;
      wait->completed = true;
      wait->ready.notify_one();
    }
    return;
  }
  if (schedule && (wait->pool == nullptr ||
                   !wait->pool->submit_recurrent(pump_stream_task, wait))) {
    const Status failure_status = Status::fail(Reason::PipelineBusy);
    residency::execution::StreamEvidence failure_final{};
    bool aborted = false;
    {
      std::lock_guard lock{wait->coordinator};
      wait->scheduled = false;
      aborted = wait->stream->abort(failure_status, pipeline_clock()) &&
                wait->stream->final(failure_final);
      wait->disposition = failure_status;
      wait->quarantine = !aborted;
      wait->terminal = true;
    }
    {
      std::lock_guard lock{wait->gate};
      wait->final = std::move(failure_final);
      wait->failure = failure_status;
      wait->poison = !aborted;
      wait->completed = true;
      wait->ready.notify_one();
    }
  }
}

} // namespace rund::compute::detail::virtual_stream_detail
