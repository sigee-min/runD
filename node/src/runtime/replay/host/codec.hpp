#pragma once

#include "../artifact/format.hpp"

#include <node/runtime/replay/host.hpp>
#include <rund/replay/limits.hpp>

#include <cstdint>
#include <span>

namespace rund::node::replay_detail {

[[nodiscard]] bool
WriteHostEvidence(artifact::Writer &out,
                  std::span<const ::rund::host::Event> events,
                  std::uint64_t event_hash) noexcept;

[[nodiscard]] HostReplayDecodeResult
ReadHostEvidence(artifact::Reader &in, std::uint64_t max_entries);

} // namespace rund::node::replay_detail
