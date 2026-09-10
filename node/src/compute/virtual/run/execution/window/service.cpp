#include "local.hpp"

#include "../../backing.hpp"
#include "../../cache.hpp"
#include "../fill.hpp"
#include "../io.hpp"

#include "../../../../device/residency/execution/run.hpp"
#include "../../../../pipeline/execution/attempt.hpp"
#include "../../../../pipeline/execution/schedule.hpp"
#include "../../../../pipeline/execution/window.hpp"
#include "../../../../pipeline/local.hpp"
#include "../../../../pipeline/run/clock.hpp"
#include "../../../backing.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <utility>

namespace rund::compute::detail {
using ::rund::detail::counter::Accumulate;

[[nodiscard]] Status ServiceWindowInput(WindowWait &wait,
                                        const std::uint64_t epoch) noexcept {
  using namespace residency::execution;
  if (wait.window == nullptr || wait.input == nullptr || wait.run == nullptr ||
      wait.stats == nullptr || epoch >= wait.input_done.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  residency::ExecutionTicket ticket{};
  if (!wait.window->issue(epoch, Phase::Input, ticket)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  return ServiceWindowInput(wait, epoch, ticket);
}

[[nodiscard]] Status
ServiceWindowInput(WindowWait &wait, const std::uint64_t epoch,
                   const residency::ExecutionTicket &ticket) noexcept {
  using namespace residency::execution;
  if (wait.window == nullptr || wait.input == nullptr || wait.run == nullptr ||
      wait.stats == nullptr || epoch >= wait.input_done.size() ||
      ticket.epoch != epoch ||
      ticket.phase != static_cast<std::uint8_t>(Phase::Input)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::size_t bank = static_cast<std::size_t>(epoch % BankCapacity);
  std::uint64_t backing_pages = 0u;
  std::uint64_t device_pages = 0u;
  Status status = execution_io::fill_input(
      *wait.input, *wait.run, ticket, wait.pipelines[bank].get(),
      wait.stats->pipeline.residency, backing_pages, &wait.input_reuse);
  if (status) {
    status = execution_io::supply_input(*wait.pipelines[bank], *wait.run,
                                        ticket, *wait.stats, device_pages);
  }
  const residency::ExecutionTerminal terminal =
      status ? residency::ExecutionTerminal::Success
             : residency::ExecutionTerminal::Failure;
  if (!wait.window->terminal(ticket, terminal, status)) {
    status = Status::fail(Reason::PipelineInvalid);
  }
  if (!status) {
    wait.failed_page = execution_io::failed_page(ticket);
    return status;
  }
  wait.input_done[static_cast<std::size_t>(epoch)] = true;
  const std::size_t count = ticket.bindings.size() / 2u;
  const residency::FrameRegion device_region = wait.run->input_regions[bank];
  std::uint64_t evictions = 0u;
  for (const residency::CacheTransition &transition : ticket.transitions) {
    evictions += static_cast<std::uint64_t>(
        transition.kind == residency::TransitionKind::Unmap &&
        transition.frame >= device_region.first &&
        transition.frame - device_region.first < device_region.count);
  }
  Accumulate(wait.stats->pipeline.residency.page_in_count, device_pages);
  Accumulate(wait.stats->pipeline.residency.cache_hit_count,
             count - device_pages);
  Accumulate(wait.stats->pipeline.residency.eviction_count, evictions);
  Accumulate(wait.stats->pipeline.residency.page_in_bytes,
             device_pages * wait.run->input_page_bytes);
  Accumulate(wait.stats->pipeline.residency.late_page_count, backing_pages);
  return Status::success();
}

namespace {

[[nodiscard]] Status ServiceWindowOutput(WindowWait &wait,
                                         const std::uint64_t epoch) noexcept {
  using namespace residency::execution;
  if (wait.window == nullptr || wait.output == nullptr || wait.run == nullptr ||
      wait.stats == nullptr || epoch >= wait.output_done.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  residency::ExecutionTicket ticket{};
  if (!wait.window->issue(epoch, Phase::Output, ticket)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::size_t index = static_cast<std::size_t>(epoch);
  const std::size_t bank = index % BankCapacity;
  Status status = retain_residency_output(*wait.pipelines[bank], *wait.run,
                                          ticket, *wait.stats);
  if (status) {
    status = execution_io::collect_output(*wait.pipelines[bank], *wait.run,
                                          ticket, wait.output_frames[index],
                                          wait.output_storage[index]);
  }
  if (status) {
    status = stage_residency_cache(*wait.output, *wait.run, ticket.transitions,
                                   wait.stats->pipeline.residency, nullptr,
                                   wait.output_frames[index]);
  }
  const residency::ExecutionTerminal terminal =
      status ? residency::ExecutionTerminal::Success
             : residency::ExecutionTerminal::Failure;
  if (!wait.window->terminal(ticket, terminal, status)) {
    status = Status::fail(Reason::PipelineInvalid);
  }
  if (!status) {
    wait.failed_page = execution_io::failed_page(ticket);
    return status;
  }
  for (std::size_t local = 0u; local < wait.output_frames[index].size();
       ++local) {
    const std::uint64_t page = epoch * wait.run->frame_capacity + local;
    const std::uint64_t logical = page * wait.run->output_payload_bytes;
    const std::size_t bytes = static_cast<std::size_t>(
        std::min(wait.run->output_payload_bytes,
                 wait.run->active.output_bytes - logical));
    wait.hash.Bytes(
        reinterpret_cast<const std::uint8_t *>(
            wait.output_frames[index][local] + wait.run->output_prefix_bytes),
        bytes);
  }
  wait.output_done[index] = true;
  return Status::success();
}

} // namespace

void CompleteWindowNative(void *const raw,
                          PipelineWindowRelease &&pipeline) noexcept {
  auto *const wait = static_cast<WindowWait *>(raw);
  if (wait == nullptr || wait->window == nullptr || wait->run == nullptr) {
    return;
  }
  std::uint64_t next = 0u;
  Status status = pipeline.release.status;
  {
    std::lock_guard lock{wait->coordinator};
    const std::uint64_t epoch = pipeline.release.epoch;
    if (!wait->window->release(pipeline.release)) {
      status = Status::fail(Reason::CompletionInvalid);
      wait->poison = true;
    }
    // Failure is a persistent window disposition, not a property of only the
    // Release that first observed it. A later already-queued native success
    // must not restart Output backing work or admit Input(e+2); every remaining
    // gate drains through Suppress after the first failure.
    if (!wait->failure && status) {
      status = wait->failure;
    }
    // Generation zero is the first valid Pipeline attempt. Publication plus
    // reseed is the exact attempt-existence proof; suppressed native drains
    // carry neither and must not fold a stale bank profile.
    if (pipeline.terminal_published && pipeline.reseeded &&
        wait->stats != nullptr) {
      const std::size_t bank =
          static_cast<std::size_t>(epoch % residency::execution::BankCapacity);
      const Status accumulated = accumulate_virtual_epoch(
          *wait->stats, pipeline_stats(wait->pipelines[bank]));
      if (!accumulated) {
        status = accumulated;
      }
    }
    if (status) {
      status = ServiceWindowOutput(*wait, epoch);
    }
    next = epoch + residency::execution::BankCapacity;
    if (status && next < wait->window->lease().epochs) {
      status = ServiceWindowInput(*wait, next);
    }
    if (!status && wait->failure) {
      wait->failure = status;
      wait->failed_page = wait->failed_page == ResidencyStats::no_failed_page
                              ? epoch * wait->run->frame_capacity
                              : wait->failed_page;
      wait->poison = wait->poison ||
                     pipeline.release.terminal ==
                         residency::execution::TerminalKind::UnknownMayWrite;
    }
  }
  if (next < wait->window->lease().epochs && wait->native != nullptr) {
    const Status signalled =
        signal_pipeline_execution_window(*wait->native, next, status);
    if (!signalled) {
      const Status aborted =
          abort_pipeline_execution_window(*wait->native, signalled);
      std::lock_guard lock{wait->coordinator};
      if (wait->failure) {
        wait->failure = aborted ? signalled : aborted;
      }
      wait->poison = true;
    }
  }
}

void CompleteWindowFinal(
    void *const raw, residency::execution::WindowEvidence &&evidence) noexcept {
  auto *const wait = static_cast<WindowWait *>(raw);
  if (wait == nullptr) {
    return;
  }
  {
    std::lock_guard lock{wait->coordinator};
    if (wait->window == nullptr || !wait->window->final(evidence)) {
      const bool abandoned =
          wait->window != nullptr &&
          evidence.terminal == residency::execution::TerminalKind::Known &&
          wait->window->abandon_final(evidence);
      evidence.status = Status::fail(Reason::CompletionInvalid);
      evidence.terminal =
          abandoned ? residency::execution::TerminalKind::Known
                    : residency::execution::TerminalKind::UnknownMayWrite;
      wait->poison = true;
    }
  }
  {
    std::lock_guard lock{wait->gate};
    wait->final = std::move(evidence);
    wait->completed = true;
    wait->ready.notify_one();
  }
}

} // namespace rund::compute::detail
