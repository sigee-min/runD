#pragma once

#include "result.hpp"
#include "state.hpp"

#include <cstddef>

namespace rund::node {

[[nodiscard]] ReactorPlatformOpResult
OpenReactorPlatform(ReactorPlatform &) noexcept;
[[nodiscard]] ReactorPlatformOpResult
PrepareReactorPlatform(ReactorPlatform &, std::size_t) noexcept;
void CloseReactorPlatform(ReactorPlatform &) noexcept;

} // namespace rund::node
