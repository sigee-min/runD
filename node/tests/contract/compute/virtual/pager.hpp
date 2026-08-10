#pragma once

#include "backing.hpp"
#include "model.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund_node_test_virtual {

// Allocation-free executable oracle for one page-local fused map. It models
// two fixed resident slots (one input, one output), not a native backend.
class MemoryVirtualPager final {
public:
  [[nodiscard]] bool run(MemoryVirtualBacking &input,
                         MemoryVirtualBacking &output,
                         std::size_t logical_elements) noexcept;
  [[nodiscard]] PagerFacts facts() const noexcept { return facts_; }

private:
  std::array<std::int32_t, PageElements> input_page_{};
  std::array<std::int32_t, PageElements> output_page_{};
  PagerFacts facts_{.resident_capacity_bytes = ResidentCapacityBytes};
};

} // namespace rund_node_test_virtual
