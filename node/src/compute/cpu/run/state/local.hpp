#pragma once

#include "../state.hpp"

namespace rund::compute::detail::cpu_run_state_detail {

[[nodiscard]] bool add_count(std::size_t &, std::size_t) noexcept;

} // namespace rund::compute::detail::cpu_run_state_detail
