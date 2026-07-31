#pragma once

#include <node/runtime/replay/record.hpp>

#include <cstdint>

namespace rund::node::replay_detail {

// Source-only path for evidence whose trace and hash were captured by the
// same trusted runtime operation. Public construction must hash its mutable
// input again instead of accepting a caller-supplied identity.
[[nodiscard]] RuntimeReplayRecord
make_runtime_replay_record(RuntimeReplayRecordDesc desc,
                           std::uint64_t trusted_trace_hash);

} // namespace rund::node::replay_detail
