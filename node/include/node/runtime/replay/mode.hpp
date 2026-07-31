#pragma once

#include <cstdint>

namespace rund::node {

namespace replay_detail {
class InputPlan;
}

enum class HostReplayMode : std::uint8_t {
  Record = 0u,
  Replay = 1u,
};

} // namespace rund::node
