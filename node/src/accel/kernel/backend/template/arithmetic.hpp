#pragma once

#include <cstdint>

namespace rund::node::accel::detail::backend_template_plan {

[[nodiscard]] bool add(std::uint64_t &target, std::uint64_t value) noexcept;

[[nodiscard]] bool product(std::uint64_t left, std::uint64_t right,
                           std::uint64_t &out) noexcept;

} // namespace rund::node::accel::detail::backend_template_plan
