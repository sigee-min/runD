#pragma once

#include "model.hpp"

#include <cstdint>

namespace rund::compute::detail::residency {

[[nodiscard]] Identity
IdentifyResidencyPlan(std::uint64_t page_bytes, std::uint64_t page_count,
                      std::uint64_t slot_capacity) noexcept;

} // namespace rund::compute::detail::residency
