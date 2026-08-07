#pragma once

#include <cstdint>

#include "model.hpp"

namespace rund::node {

[[nodiscard]] bool
ReactorRegistrationCollectForWaitAdd(ReactorRuntime &reactor, ReactorHandle fd,
                                     ReactorInterest current_interest,
                                     std::uint64_t fd_generation) noexcept;

[[nodiscard]] bool ReactorRegistrationCollectForWaitRemove(
    ReactorRuntime &reactor, ReactorHandle fd,
    ReactorInterest previous_interest,
    ReactorInterest current_interest) noexcept;

[[nodiscard]] bool
ReactorRegistrationFlushDeferredRemoves(ReactorRuntime &reactor) noexcept;

[[nodiscard]] bool
ReactorRegistrationHasDeferredRemoves(const ReactorRuntime &reactor) noexcept;

void ReactorRegistrationForgetGeneration(
    ReactorRuntime &reactor, ReactorHandle fd,
    std::uint64_t fd_generation) noexcept;

void ReactorRegistrationClear(ReactorRuntime &reactor) noexcept;

} // namespace rund::node
