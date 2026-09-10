#include "internal.hpp"

#include "../../../../pipeline/run/clock.hpp"

#include <algorithm>
#include <thread>

namespace rund::compute::detail::virtual_stream_detail {

void pump_stream_task(void *const raw) noexcept {
  auto *const wait = static_cast<StreamWait *>(raw);
  if (wait != nullptr) {
    pump_stream(*wait);
  }
}

void pump_stream(StreamWait &wait) noexcept {
  using namespace residency::execution;
  {
    std::lock_guard lock{wait.coordinator};
    wait.callback_on_caller =
        wait.callback_on_caller || std::this_thread::get_id() == wait.caller;
    wait.scheduled = false;
    if (wait.pumping || wait.terminal || wait.plan == nullptr ||
        wait.stream == nullptr || wait.native == nullptr ||
        wait.claim == nullptr) {
      return;
    }
    wait.pumping = true;
  }
  for (;;) {
    std::uint64_t first = 0u;
    std::size_t count = 0u;
    {
      std::lock_guard lock{wait.coordinator};
      if (!wait.pending || wait.terminal) {
        wait.pumping = false;
        return;
      }
      wait.pending = false;
      first = wait.chunk_first + wait.chunk_count;
      count = static_cast<std::size_t>(std::min<std::uint64_t>(
          WindowCapacity, wait.stream->lease().epochs - first));
      wait.chunk_first = first;
      wait.chunk_count = count;
    }
    Status status = rearm_pipeline_execution_window(*wait.native);
    if (status) {
      status = submit_pipeline_execution_stream_chunk(
          *wait.plan, wait.stream->lease(), first, count, wait.pipelines,
          complete_stream_native, complete_stream_final, &wait, *wait.native,
          *wait.claim);
    }
    if (!status) {
      StreamEvidence final{};
      bool aborted = false;
      {
        std::lock_guard lock{wait.coordinator};
        wait.disposition = status;
        aborted = wait.stream->abort(status, pipeline_clock()) &&
                  wait.stream->final(final);
        wait.quarantine = !aborted;
        wait.terminal = true;
        wait.pumping = false;
      }
      {
        std::lock_guard lock{wait.gate};
        wait.final = std::move(final);
        wait.failure = status;
        wait.poison = !aborted;
        wait.completed = true;
        wait.ready.notify_one();
      }
      return;
    }
    const std::uint64_t bootstrap =
        std::min<std::uint64_t>(count, BankCapacity);
    for (std::uint64_t slot = 0u; slot < bootstrap; ++slot) {
      const std::uint64_t epoch = first + slot;
      Status admission = Status::success();
      {
        std::lock_guard lock{wait.coordinator};
        if (!wait.disposition) {
          // A prior chunk bootstrap failed after this native chunk had been
          // accepted. Consume the remaining exact Host-service tickets as
          // failures without issuing more backing I/O, then open their gates
          // through Suppress so the accepted suffix drains deterministically.
          admission = wait.disposition;
          residency::ExecutionTicket ticket{};
          if (!wait.stream->issue(epoch, Phase::Input, ticket) ||
              !wait.stream->terminal(
                  ticket, residency::ExecutionTerminal::Failure, admission)) {
            admission = Status::fail(Reason::PipelineInvalid);
            wait.quarantine = true;
          }
        } else {
          admission = service_stream_input(wait, epoch);
        }
        if (!admission && wait.disposition) {
          wait.disposition = admission;
        }
      }
      const Status signalled = signal_stream_native(wait, epoch, admission);
      if (!signalled) {
        const Status aborted = abort_stream_native(wait, signalled);
        std::lock_guard lock{wait.coordinator};
        wait.disposition = aborted ? signalled : aborted;
        wait.quarantine = true;
        break;
      }
    }
  }
}

} // namespace rund::compute::detail::virtual_stream_detail
