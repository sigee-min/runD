#pragma once

#include <cstdint>
#include <span>

namespace rund_node_test_virtual::product {

[[nodiscard]] std::int32_t SeedValue(std::size_t index) noexcept;
[[nodiscard]] std::int32_t ProductValue(std::int32_t value) noexcept;
void SeedInput(std::span<std::int32_t> values) noexcept;
[[nodiscard]] bool GoldenMatches(std::span<const std::int32_t> values) noexcept;
[[nodiscard]] std::uint64_t
HashValues(std::span<const std::int32_t> values) noexcept;

} // namespace rund_node_test_virtual::product
