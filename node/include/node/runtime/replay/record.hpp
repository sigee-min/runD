#pragma once

#include <node/runtime/replay/check.hpp>
#include <node/runtime/replay/hash.hpp>
#include <node/runtime/replay/model.hpp>

#include <string_view>
#include <vector>

namespace rund::node {

namespace replay_detail {
inline constexpr task::Stats kDefaultRecordTasks{};
}

// One named, consuming construction boundary for native runtime replay
// evidence. The task snapshot is borrowed for this call and copied exactly
// once into the resulting record. Moved fields transfer their
// default-allocator storage into that record.
struct RuntimeReplayRecordDesc {
  std::uint64_t start_hash = replay_detail::GenesisStartHash();
  ::rund::replay::Code code = ::rund::replay::Code::SessionResultMissing;
  const task::Stats &tasks = replay_detail::kDefaultRecordTasks;
  std::vector<task::Observation> observations{};
  std::vector<::rund::host::Event> host_events{};
  ::rund::node::replay_detail::payload::Archive host_payload_archive{};
  ::rund::Trace trace{};
};

[[nodiscard]] RuntimeReplayRecord
make_runtime_replay_record(RuntimeReplayRecordDesc desc);

[[nodiscard]] bool
valid_runtime_replay_record(const RuntimeReplayRecord &record);

[[nodiscard]] RuntimeReplayCheck
check_runtime_replay(const RuntimeReplayRecord &expected,
                     const RuntimeReplayRecord &actual) noexcept;

} // namespace rund::node
