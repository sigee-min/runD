#include "window/local.hpp"

#include "../cache.hpp"
#include "../execution.hpp"

#include "../../../device/residency/execution/run.hpp"
#include "../../../device/residency/execution/sliding.hpp"
#include "../../../device/residency/execution/stream.hpp"
#include "../../../device/residency/pool.hpp"
#include "../../../pipeline/execution/attempt.hpp"
#include "../../../pipeline/execution/schedule.hpp"
#include "../../../pipeline/execution/submit.hpp"
#include "../../../pipeline/execution/window.hpp"
#include "../../../pipeline/local.hpp"
#include "../../../pipeline/run/clock.hpp"
#include "../../../pipeline/state.hpp"
#include "../../backing.hpp"
#include "../../stats.hpp"

#include <rund/compute/pipeline/runtime.hpp>
#include <rund/counter.hpp>

#include <algorithm>
#include <array>
#include <mutex>

namespace rund::compute::detail {
using ::rund::detail::counter::Accumulate;

[[nodiscard]] VirtualExecutionResult
execute_virtual_execution_stream(VirtualPipelineState &, VirtualBacking &,
                                 VirtualBacking &, const VirtualRunProjection &,
                                 const VirtualExecutionWindowPrepared &,
                                 Stats &) noexcept;

VirtualExecutionResult execute_virtual_execution_window(
    VirtualPipelineState &state, VirtualBacking &input, VirtualBacking &output,
    const VirtualRunProjection &run,
    const VirtualExecutionWindowPrepared &prepared, Stats &stats) noexcept {
  using namespace residency;
  using namespace residency::execution;
  VirtualExecutionResult result{};
  // This bounded controller owns one cold serial Host-service continuation.
  // A backing that explicitly exposes parallel read service retains the
  // rolling two-lane prefetch path until the execution controller owns two
  // authenticated Host-service lanes; serializing it here would be a real
  // throughput regression hidden behind a GPU-driven label.
  if (input.max_parallel_reads() > 1u) {
    result.status = Status::fail(Reason::BackendUnsupported);
    return result;
  }
  if (prepared.plan.epoch_count() > WindowCapacity) {
    return execute_virtual_execution_stream(state, input, output, run, prepared,
                                            stats);
  }
  if (!prepared || state.pipeline == nullptr ||
      state.alternate_pipeline == nullptr ||
      state.pipeline->residency_pool == nullptr) {
    return result;
  }
  Pool &pool = *state.pipeline->residency_pool;
  Window joined{};
  const ExecutionLease lease = joined.begin(pool.authority(), prepared.plan);
  if (!lease) {
    result.status = Status::fail(lease.failure == AuthorityFailure::Busy
                                     ? Reason::PipelineBusy
                                     : Reason::PipelineMemoryBudget);
    return result;
  }

  PipelineResidencyWindowControl native{};
  WindowWait wait{
      .window = &joined,
      .native = &native,
      .state = &state,
      .input = &input,
      .output = &output,
      .run = &run,
      .stats = &stats,
      .pipelines = prepared.pipelines,
  };
  // Authenticate both initially live banks before the one native handoff.
  // A warm cache permutation can make a fixed-local window infeasible; that
  // must fall back with zero native submissions instead of accepting a queue
  // whose first ready gate can never be opened.
  const std::uint64_t bootstrap =
      std::min<std::uint64_t>(lease.epochs, BankCapacity);
  std::array<residency::ExecutionTicket, BankCapacity> bootstrap_tickets{};
  for (std::uint64_t epoch = 0u; epoch < bootstrap; ++epoch) {
    if (!joined.issue(epoch, Phase::Input,
                      bootstrap_tickets[static_cast<std::size_t>(epoch)])) {
      const bool abandoned = joined.abandon();
      result.status = Status::fail(abandoned ? Reason::BackendUnsupported
                                             : Reason::PipelineInvalid);
      result.poison_pipeline = !abandoned;
      return result;
    }
  }
  const std::uint64_t submitted_ns = pipeline_clock();
  Status status = submit_pipeline_execution_window(
      prepared.plan, lease, prepared.pipelines, CompleteWindowNative,
      CompleteWindowFinal, &wait, native);
  if (!status) {
    const bool abandoned = joined.abandon();
    result.status = status;
    result.poison_pipeline = !abandoned;
    return result;
  }

  // Bootstrap is fixed in physical bank count rather than epoch count. The
  // main thread performs no epoch loop: later Input/Output services are driven
  // by internal native bank Releases and the run waits only on one Final.
  for (std::uint64_t epoch = 0u; epoch < bootstrap; ++epoch) {
    Status admission = Status::success();
    {
      // Signal(e0) may complete immediately on another thread while the main
      // thread is authenticating e1. Window, service tickets, Stats, and the
      // persistent failure disposition have one coordinator; none of them is
      // internally thread-safe. Keep the fixed two-bank bootstrap under that
      // same authority, but never hold it across a backend Signal because a
      // valid backend may deliver Release inline.
      std::lock_guard lock{wait.coordinator};
      const residency::ExecutionTicket &ticket =
          bootstrap_tickets[static_cast<std::size_t>(epoch)];
      if (!wait.failure) {
        // Keep the already-issued second-bank service terminal exact, but do
        // not run backing reads after the first bootstrap bank failed. The
        // accepted native suffix drains through its no-write gate.
        admission = wait.failure;
        if (!joined.terminal(ticket, ExecutionTerminal::Failure, admission)) {
          admission = Status::fail(Reason::PipelineInvalid);
          wait.poison = true;
        }
      } else {
        admission = ServiceWindowInput(wait, epoch, ticket);
      }
      if (!admission && wait.failure) {
        wait.failure = admission;
      }
    }
    const Status signalled =
        signal_pipeline_execution_window(native, epoch, admission);
    if (!signalled) {
      {
        std::lock_guard lock{wait.coordinator};
        wait.poison = true;
        if (wait.failure) {
          wait.failure = signalled;
        }
      }
      // Execute and suppress both failed after one native handoff was already
      // accepted. Force the exact-generation backend drain: every unopened
      // gate becomes UnknownMayWrite, the prepared owners are quarantined,
      // and the one Final callback wakes this join. A timeout-success or a
      // caller-side lease rollback would expose may-write bytes.
      const Status aborted = abort_pipeline_execution_window(native, signalled);
      {
        std::lock_guard lock{wait.coordinator};
        if (!aborted && wait.failure) {
          wait.failure = aborted;
        }
      }
      break;
    }
  }

  {
    std::unique_lock lock{wait.gate};
    wait.ready.wait(lock, [&wait] { return wait.completed; });
  }
  Accumulate(stats.pipeline.residency.stall_ns,
             pipeline_clock() - submitted_ns);
  if (!wait.final.status && wait.failure) {
    wait.failure = wait.final.status;
  }
  if (!joined.snapshot().final) {
    result.status = Status::fail(Reason::CompletionInvalid);
    result.poison_pipeline = true;
    return result;
  }
  const ExecutionClose closed = joined.close();
  if (!closed) {
    result.status = Status::fail(Reason::PipelineInvalid);
    result.poison_pipeline = true;
    return result;
  }
  // The accepted Final is the sole producer for the public window receipt.
  // Record it before folding success so a backend-authenticated failed Final
  // remains distinguishable from Q1, rolling, and pre-native fallback.
  Accumulate(stats.pipeline.residency.window_handoff_count,
             wait.final.public_handoffs);
  Accumulate(stats.pipeline.residency.window_batch_count,
             wait.final.native_batches);
  Accumulate(stats.pipeline.residency.window_queue_call_count,
             wait.final.queue_calls);
  stats.command_submits = wait.final.queue_calls;
  if (!closed.success || !wait.failure) {
    result.status = wait.failure ? wait.final.status : wait.failure;
    result.failed_page = wait.failed_page;
    result.poison_pipeline =
        wait.poison || wait.final.terminal == TerminalKind::UnknownMayWrite;
    return result;
  }

  publish_residency_cache(output);
  clear_virtual_recovery(output);
  Accumulate(stats.pipeline.residency.epoch_count, lease.epochs);
  result.status = Status::success();
  result.output_hash = wait.hash.Finish();
  return result;
}

} // namespace rund::compute::detail
