#pragma once

#include <node/runtime/replay/host/archive.hpp>
#include <rund/host/event.hpp>

#include <cstdint>
#include <vector>

namespace rund::node {

struct HostReplayEvidence {
  std::vector<::rund::host::Event> events{};
  std::uint64_t event_hash = 0u;
  ::rund::node::replay_detail::payload::Archive payload_archive{};
};

} // namespace rund::node
