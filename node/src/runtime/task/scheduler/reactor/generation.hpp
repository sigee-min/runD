#pragma once

#include <rund/task/handle.hpp>

#include <cstdint>
#include <vector>

#include "model.hpp"

namespace rund::node {

class Scheduler;

[[nodiscard]] bool ReactorGenerationCollectStaleWaits(
    const ReactorRuntime &reactor, ReactorHandle fd,
    std::uint64_t current_generation, std::vector<ReactorWait> &stale) noexcept;

[[nodiscard]] ReasonCode
ReactorGenerationCleanupStaleWaits(Scheduler &scheduler, ReactorHandle fd,
                                   std::uint64_t current_generation) noexcept;

[[nodiscard]] bool ReactorGenerationCleanupInvalidWaits(
    Scheduler &scheduler, bool *invalidated) noexcept;

} // namespace rund::node
