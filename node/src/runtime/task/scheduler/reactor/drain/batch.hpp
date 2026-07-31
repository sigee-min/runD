#pragma once

#include <vector>

#include "../model.hpp"

namespace rund::node {

struct ReactorDrainBatch {
  const std::vector<ReactorReady>* ready = nullptr;
  const std::vector<ReactorWait>* removed_waits = nullptr;
  bool ok = true;
};

[[nodiscard]] ReactorDrainBatch ReactorBuildDrainBatch(
    ReactorRuntime& reactor,
    const std::vector<ReactorReady>& ordered) noexcept;

}  // namespace rund::node
