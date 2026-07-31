#pragma once

#include <rund/task/result.hpp>

#include <string>
#include <string_view>

namespace rund::host::env {

[[nodiscard]] task::Result<std::string> get(std::string_view name) noexcept;

} // namespace rund::host::env
