#include "pager.hpp"

#include "golden.hpp"

#include <algorithm>
#include <span>

namespace rund_node_test_virtual {

bool MemoryVirtualPager::run(MemoryVirtualBacking &input,
                             MemoryVirtualBacking &output,
                             const std::size_t logical_elements) noexcept {
  if (logical_elements > input.size_bytes() / sizeof(std::int32_t) ||
      logical_elements > output.size_bytes() / sizeof(std::int32_t)) {
    return false;
  }
  const std::size_t pages = logical_elements / PageElements +
                            (logical_elements % PageElements == 0u ? 0u : 1u);
  for (std::size_t page = 0u; page < pages; ++page) {
    const std::size_t first = page * PageElements;
    const std::size_t count = std::min(PageElements, logical_elements - first);
    const std::size_t bytes = count * sizeof(std::int32_t);
    auto input_values = std::span<std::int32_t>{input_page_}.first(count);
    auto output_values = std::span<std::int32_t>{output_page_}.first(count);
    facts_.resident_current_bytes = ResidentCapacityBytes;
    facts_.resident_peak_bytes =
        std::max(facts_.resident_peak_bytes, facts_.resident_current_bytes);
    if (!input.load(first * sizeof(std::int32_t),
                    std::as_writable_bytes(input_values))) {
      facts_.resident_current_bytes = 0u;
      return false;
    }
    for (std::size_t local = 0u; local < count; ++local) {
      output_values[local] = FusedValue(input_values[local], first + local);
    }
    if (!output.store(first * sizeof(std::int32_t),
                      std::as_bytes(output_values).first(bytes))) {
      facts_.resident_current_bytes = 0u;
      return false;
    }
    ++facts_.epoch_count;
    facts_.resident_current_bytes = 0u;
  }
  ++facts_.run_count;
  return true;
}

} // namespace rund_node_test_virtual
