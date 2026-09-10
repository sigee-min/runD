#pragma once

#include "../prepared.hpp"

#include <cstddef>
#include <cstdint>

namespace rund::compute::detail::cpu_prepared_detail {

[[nodiscard]] bool
valid_tile_plan(const kernel::ComputeTileRunStoragePlan &) noexcept;
[[nodiscard]] bool has_execution(const CpuExecutionStoragePlan &) noexcept;
[[nodiscard]] std::uint64_t saturating_extent(std::size_t count,
                                              std::size_t width) noexcept;
[[nodiscard]] std::uint64_t saturating_add(std::uint64_t,
                                           std::uint64_t) noexcept;

} // namespace rund::compute::detail::cpu_prepared_detail
