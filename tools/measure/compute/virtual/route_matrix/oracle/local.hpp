#pragma once

#include "../internal.hpp"

namespace rund::measure::compute::route_matrix::oracle::oracle_detail {

[[nodiscard]] std::int32_t window_value(std::uint64_t index,
                                        std::uint64_t count,
                                        std::uint64_t radius) noexcept;
[[nodiscard]] bool mul(std::uint64_t left, std::uint64_t right,
                       std::uint64_t &value) noexcept;
[[nodiscard]] bool device_vsm_cohort(const Row &row) noexcept;
[[nodiscard]] bool window_ring_cohort(const Row &row) noexcept;
[[nodiscard]] bool memory_receipt_ok(const Row &row) noexcept;
[[nodiscard]] bool device_vsm_proof_ok(const Row &row) noexcept;

} // namespace rund::measure::compute::route_matrix::oracle::oracle_detail
