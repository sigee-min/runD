#pragma once

#include "../model/frame.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace rund::compute::detail::residency::lease_detail {

[[nodiscard]] std::size_t
select_frame(const std::vector<registry_model::Frame> &,
             std::span<const CacheUse>, std::size_t first, std::size_t count,
             std::uint64_t epoch) noexcept;

[[nodiscard]] TransitionKind retire_dirty(CacheKey) noexcept;

} // namespace rund::compute::detail::residency::lease_detail
