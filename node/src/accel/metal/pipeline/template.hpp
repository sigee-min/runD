#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

// Frozen primitive executable tuple. The five-slot ceiling is the audited
// maximum for one Metal primitive (Sort, Compact, and Partition). Route-owned
// buffers and mutable dispatch state never enter this Program-level owner.
struct MetalKernelImmutablePipelines final {
  std::array<std::shared_ptr<void>, 5u> stages{};
  std::uint32_t count{};

  [[nodiscard]] bool ready(const std::uint32_t expected) const noexcept {
    if (count != expected || count == 0u || count > stages.size()) {
      return false;
    }
    for (std::size_t index = 0u; index < count; ++index) {
      if (stages[index] == nullptr) {
        return false;
      }
    }
    return true;
  }
};

} // namespace rund::node::accel::detail
