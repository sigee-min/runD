#pragma once

#include "../evidence.hpp"

namespace rund_node_test_virtual::product {

[[nodiscard]] bool
ProductExecutionCommonMatches(const ProductExecutionEvidence &) noexcept;
[[nodiscard]] bool
ProductExecutionRollingMatches(const ProductExecutionEvidence &) noexcept;
[[nodiscard]] bool
ProductExecutionPersistentMatches(const ProductExecutionEvidence &) noexcept;
[[nodiscard]] bool
ProductExecutionDeviceVsmMatches(const ProductExecutionEvidence &) noexcept;
[[nodiscard]] bool
ProductExecutionWindowMatches(const ProductExecutionEvidence &) noexcept;

} // namespace rund_node_test_virtual::product
