#include <rund/telemetry/event.hpp>

#include <rund/counter.hpp>

namespace rund::telemetry {

Findings Event::findings() const noexcept {
  Findings out{};
  if (source == Source::Compute) {
    if (compute.buffer_allocations != 0u) {
      (void)out.append(Finding{
          .cost = Cost::Allocation,
          .unit = Unit::Events,
          .accuracy = accuracy(compute.buffer_allocations),
          .reference_accuracy = accuracy(compute.buffer_reuses),
          .reference_kind = Reference::ReuseEvents,
          .cause = Cause::BufferAllocation,
          .action = Action::ReuseJob,
          .observed = compute.buffer_allocations,
          .reference = compute.buffer_reuses,
      });
    }
    if (compute.copied_bytes != 0u) {
      (void)out.append(Finding{
          .cost = Cost::Copy,
          .unit = Unit::Bytes,
          .accuracy = accuracy(compute.copied_bytes),
          .cause = Cause::BoundaryCopy,
          .action = Action::KeepResident,
          .observed = compute.copied_bytes,
      });
    }
    if (compute.graph_read_bytes != 0u) {
      (void)out.append(Finding{
          .cost = Cost::Scan,
          .unit = Unit::Bytes,
          .accuracy = accuracy(compute.graph_read_bytes),
          .cause = Cause::GraphRead,
          .action = Action::ReduceGraphBound,
          .observed = compute.graph_read_bytes,
      });
    }
  } else if (source == Source::Replay) {
    if (replay.storage_growths != 0u) {
      (void)out.append(Finding{
          .cost = Cost::Allocation,
          .unit = Unit::Events,
          .accuracy = accuracy(replay.storage_growths),
          .cause = Cause::StorageGrowth,
          .action = Action::ConfigureStorage,
          .observed = replay.storage_growths,
      });
    }
    if (replay.copied_bytes != 0u) {
      (void)out.append(Finding{
          .cost = Cost::Copy,
          .unit = Unit::Bytes,
          .accuracy = accuracy(replay.copied_bytes),
          .reference_accuracy = accuracy(replay.retained_bytes),
          .reference_kind = Reference::RetainedBytes,
          .cause = Cause::ReplayCopy,
          .action = Action::ConfigureStorage,
          .observed = replay.copied_bytes,
          .reference = replay.retained_bytes,
      });
    }
  }
  if (queue.capacity != 0u && queue.depth >= queue.capacity) {
    (void)out.append(Finding{
        .cost = Cost::Queue,
        .unit = Unit::Entries,
        .accuracy = accuracy(queue.depth),
        .reference_accuracy = accuracy(queue.capacity),
        .reference_kind = Reference::QueueCapacity,
        .cause = Cause::QueueAtBound,
        .action = Action::ReduceFanout,
        .observed = queue.depth,
        .reference = queue.capacity,
    });
  }

  if (level == Level::Basic) {
    (void)out.append(Finding{
        .cost = Cost::CriticalPath,
        .unit = Unit::Nanoseconds,
        .accuracy = Accuracy::Unavailable,
        .cause = Cause::TimingUnavailable,
        .action = Action::EnableDetail,
    });
    return out;
  }

  const std::uint64_t maximum =
      detail.prepare_ns > detail.work_ns
          ? (detail.prepare_ns > detail.finish_ns ? detail.prepare_ns
                                                  : detail.finish_ns)
          : (detail.work_ns > detail.finish_ns ? detail.work_ns
                                               : detail.finish_ns);
  Cause cause = Cause::None;
  Action action = Action::None;
  if (detail.prepare_ns == maximum) {
    cause |= Cause::Prepare;
    action |= source == Source::Replay ? Action::ReuseReplayPlan
                                       : Action::ReuseProgram;
  }
  if (detail.work_ns == maximum) {
    cause |= Cause::Work;
    if (source == Source::Replay) {
      action |= Action::ReduceReplayEvidence;
    } else if (compute.command_submits != 0u &&
               compute.kernel_samples != 0u &&
               compute.submit_wait_ns > compute.kernel_ns) {
      cause |= Cause::SubmitOverhead;
      action |= Action::BatchJobs;
    } else {
      action |= Action::ReduceGraphBound;
    }
  }
  if (detail.finish_ns == maximum) {
    cause |= Cause::Finish;
    action |= source == Source::Replay ? Action::ReduceReplayEvidence
                                       : Action::ReadSelectedOutput;
  }
  const std::uint64_t total = ::rund::detail::counter::SaturatingAdd(
      ::rund::detail::counter::SaturatingAdd(detail.prepare_ns, detail.work_ns),
      detail.finish_ns);
  (void)out.append(Finding{
      .cost = Cost::CriticalPath,
      .unit = Unit::Nanoseconds,
      .accuracy = accuracy(maximum),
      .reference_accuracy = accuracy(total),
      .reference_kind = Reference::PhaseTotal,
      .cause = cause,
      .action = action,
      .observed = maximum,
      .reference = total,
  });
  return out;
}

} // namespace rund::telemetry
