#pragma once

#include <cstdint>

namespace package_consumer::allocation {

void start() noexcept;
[[nodiscard]] std::uint64_t stop() noexcept;

} // namespace package_consumer::allocation
