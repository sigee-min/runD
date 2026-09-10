#pragma once

#include "../store.hpp"

namespace rund::node::replay_detail::payload::store_detail {

[[nodiscard]] bool fits_budget(std::uint64_t current, std::uint64_t added,
                               std::uint64_t budget) noexcept;
[[nodiscard]] bool fits_u32(std::size_t value) noexcept;

} // namespace rund::node::replay_detail::payload::store_detail
