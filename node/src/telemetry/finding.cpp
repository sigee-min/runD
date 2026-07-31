#include <rund/telemetry/finding.hpp>

namespace rund::telemetry {

std::string_view name(const Cost value) noexcept {
  switch (value) {
  case Cost::Allocation:
    return "allocation";
  case Cost::Copy:
    return "copy";
  case Cost::Scan:
    return "scan";
  case Cost::Queue:
    return "queue";
  case Cost::CriticalPath:
    return "critical-path";
  }
  return "invalid";
}

std::string_view name(const Unit value) noexcept {
  switch (value) {
  case Unit::Events:
    return "events";
  case Unit::Bytes:
    return "bytes";
  case Unit::Entries:
    return "entries";
  case Unit::Nanoseconds:
    return "ns";
  }
  return "invalid";
}

std::string_view name(const Accuracy value) noexcept {
  switch (value) {
  case Accuracy::Exact:
    return "exact";
  case Accuracy::Unavailable:
    return "unavailable";
  case Accuracy::Saturated:
    return "saturated";
  }
  return "invalid";
}

std::string_view name(const Reference value) noexcept {
  switch (value) {
  case Reference::None:
    return "none";
  case Reference::ReuseEvents:
    return "reuse-events";
  case Reference::RetainedBytes:
    return "retained-bytes";
  case Reference::QueueCapacity:
    return "queue-capacity";
  case Reference::PhaseTotal:
    return "phase-total";
  }
  return "invalid";
}

std::string_view name(const Cause value) noexcept {
  switch (value) {
  case Cause::None:
    return "none";
  case Cause::BufferAllocation:
    return "buffer-allocation";
  case Cause::StorageGrowth:
    return "storage-growth";
  case Cause::BoundaryCopy:
    return "boundary-copy";
  case Cause::ReplayCopy:
    return "replay-copy";
  case Cause::GraphRead:
    return "graph-read";
  case Cause::QueueAtBound:
    return "queue-at-bound";
  case Cause::TimingUnavailable:
    return "timing-unavailable";
  case Cause::Prepare:
    return "prepare";
  case Cause::Work:
    return "work";
  case Cause::Finish:
    return "finish";
  case Cause::SubmitOverhead:
    return "submit-overhead";
  }
  return "multiple";
}

std::string_view name(const Action value) noexcept {
  switch (value) {
  case Action::None:
    return "none";
  case Action::ReuseJob:
    return "reuse-job";
  case Action::ConfigureStorage:
    return "configure-storage";
  case Action::KeepResident:
    return "keep-resident";
  case Action::ReduceGraphBound:
    return "reduce-graph-bound";
  case Action::ReduceFanout:
    return "reduce-fanout";
  case Action::EnableDetail:
    return "enable-detail";
  case Action::ReuseProgram:
    return "reuse-program";
  case Action::ReadSelectedOutput:
    return "read-selected-output";
  case Action::ReuseReplayPlan:
    return "reuse-replay-plan";
  case Action::ReduceReplayEvidence:
    return "reduce-replay-evidence";
  case Action::BatchJobs:
    return "batch-jobs";
  }
  return "multiple";
}

} // namespace rund::telemetry
