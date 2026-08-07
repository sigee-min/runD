#pragma once

#include <span>

#include "model.hpp"

namespace rund::node {

[[nodiscard]] bool ReactorExpandPlatformReady(
    ReactorRuntime &reactor,
    std::span<const ReactorPlatformReady> ready) noexcept;
[[nodiscard]] bool ReactorExpandInvalidHandle(ReactorRuntime &reactor,
                                              ReactorHandle handle) noexcept;
[[nodiscard]] bool ReactorExpandPollFailure(ReactorRuntime &reactor) noexcept;
[[nodiscard]] bool ReactorExpandInvalidAll(ReactorRuntime &reactor) noexcept;

} // namespace rund::node
