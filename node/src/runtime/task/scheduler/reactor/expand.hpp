#pragma once

#include "model.hpp"

namespace rund::node {

[[nodiscard]] bool
ReactorExpandPlatformReady(ReactorRuntime &reactor,
                           const ReactorPlatformPollResult &poll) noexcept;
[[nodiscard]] bool ReactorExpandInvalidHandle(ReactorRuntime &reactor,
                                              ReactorHandle handle) noexcept;
[[nodiscard]] bool ReactorExpandPollFailure(ReactorRuntime &reactor) noexcept;
[[nodiscard]] bool ReactorExpandInvalidAll(ReactorRuntime &reactor) noexcept;

} // namespace rund::node
