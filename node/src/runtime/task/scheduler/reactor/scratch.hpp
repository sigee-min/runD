#pragma once

#include <rund/host/event.hpp>

#include <cstddef>
#include <vector>

#include "model.hpp"

namespace rund::node {

[[nodiscard]] bool
ReactorScratchPreparePlatformReady(ReactorRuntime &reactor,
                                   std::size_t reactor_capacity) noexcept;

[[nodiscard]] bool
ReactorScratchOrderReady(ReactorRuntime &reactor,
                         const std::vector<ReactorReady> &ready) noexcept;

[[nodiscard]] bool
ReactorScratchPrepareHostEvents(std::vector<::rund::host::Event> &scratch,
                                std::size_t capacity) noexcept;

void ReactorScratchClear(ReactorRuntime &reactor) noexcept;

} // namespace rund::node
