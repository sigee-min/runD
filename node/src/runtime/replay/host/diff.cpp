#include <node/runtime/replay/host.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rund::node::replay_detail {

bool HostReplayEventsEqual(const ::rund::host::Event& expected,
                           const ::rund::host::Event& actual) noexcept {
  return expected.sequence == actual.sequence && expected.kind == actual.kind &&
         expected.status == actual.status &&
         expected.task_id == actual.task_id &&
         expected.logical_time_ns == actual.logical_time_ns &&
         expected.stream_id == actual.stream_id &&
         expected.draw_id == actual.draw_id &&
         expected.host_handle_id == actual.host_handle_id &&
         expected.offset == actual.offset &&
         expected.requested_bytes == actual.requested_bytes &&
         expected.completed_bytes == actual.completed_bytes &&
         expected.native_errno == actual.native_errno &&
         expected.name_hash.value == actual.name_hash.value &&
         expected.path_hash.value == actual.path_hash.value &&
         expected.payload_hash.value == actual.payload_hash.value;
}

void AppendHostReplayWindow(std::vector<::rund::host::Event>& out,
                            const std::vector<::rund::host::Event>& events,
                            const std::size_t center,
                            const std::size_t context) {
  if (events.empty()) {
    return;
  }
  const std::size_t bounded_center =
      center < events.size() ? center : events.size() - 1u;
  const std::size_t begin =
      bounded_center > context ? bounded_center - context : 0u;
  const std::size_t available_right = events.size() - bounded_center - 1u;
  const std::size_t right =
      context < available_right ? context : available_right;
  const std::size_t end = bounded_center + right + 1u;
  out.insert(out.end(),
             events.begin() + static_cast<std::ptrdiff_t>(begin),
             events.begin() + static_cast<std::ptrdiff_t>(end));
}

bool FindFirstHostReplayEventMismatch(
    const std::vector<::rund::host::Event>& expected,
    const std::vector<::rund::host::Event>& actual,
    std::size_t& index) noexcept {
  const std::size_t common =
      expected.size() < actual.size() ? expected.size() : actual.size();
  for (std::size_t current = 0u; current < common; ++current) {
    if (!HostReplayEventsEqual(expected[current], actual[current])) {
      index = current;
      return true;
    }
  }
  if (expected.size() != actual.size()) {
    index = common;
    return true;
  }
  return false;
}

HostReplayFieldDiff DiffHostReplayEvidence(
    const HostReplayEvidence& expected,
    const HostReplayEvidence& actual) {
  if (expected.events.size() != actual.events.size()) {
    return HostReplayFieldDiff{
        .mismatch = true,
        .field = "host.count",
        .expected = static_cast<std::uint64_t>(expected.events.size()),
        .actual = static_cast<std::uint64_t>(actual.events.size()),
    };
  }
  std::size_t index = 0u;
  if (FindFirstHostReplayEventMismatch(expected.events, actual.events, index)) {
    return HostReplayFieldDiff{
        .mismatch = true,
        .field = "host.detail",
        .expected = ::rund::host::hash_event(expected.events[index]).value,
        .actual = ::rund::host::hash_event(actual.events[index]).value,
    };
  }
  return HostReplayFieldDiff{};
}

}  // namespace rund::node::replay_detail
