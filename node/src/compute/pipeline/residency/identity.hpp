#pragma once

#include "model.hpp"

#include <cstdint>
#include <vector>

namespace rund::compute::detail::residency {

[[nodiscard]] Identity
IdentifyResidencyPlan(std::uint64_t page_bytes, std::uint64_t page_count,
                      std::uint64_t frame_capacity) noexcept;

[[nodiscard]] Identity
IdentifyResidencyPlan(std::uint64_t page_bytes, std::uint32_t frame_capacity,
                      const std::vector<PageUse> &uses,
                      const std::vector<Epoch> &epochs) noexcept;

} // namespace rund::compute::detail::residency
