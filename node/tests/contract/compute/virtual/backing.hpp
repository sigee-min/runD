#pragma once

#include "model.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace rund_node_test_virtual {

// Test-only logical backing authority. This deliberately does not derive from
// the moving public VirtualBacking surface: the product bridge must be a
// separate leaf once that ABI and its production implementation are stable.
class MemoryVirtualBacking final {
public:
  explicit MemoryVirtualBacking(std::size_t bytes,
                                std::byte initial = TailPoison);

  [[nodiscard]] bool seed(std::size_t offset,
                          std::span<const std::byte> input) noexcept;
  [[nodiscard]] bool load(std::size_t offset,
                          std::span<std::byte> output) noexcept;
  [[nodiscard]] bool store(std::size_t offset,
                           std::span<const std::byte> input) noexcept;
  [[nodiscard]] bool observe(std::size_t offset,
                             std::span<std::byte> output) noexcept;
  [[nodiscard]] bool tail_is(std::size_t offset,
                             std::byte expected) const noexcept;

  [[nodiscard]] std::size_t size_bytes() const noexcept {
    return bytes_.size();
  }
  [[nodiscard]] const void *identity() const noexcept { return bytes_.data(); }
  [[nodiscard]] BackingFacts facts() const noexcept { return facts_; }

private:
  [[nodiscard]] bool contains(std::size_t offset,
                              std::size_t bytes) const noexcept;

  std::vector<std::byte> bytes_;
  BackingFacts facts_{};
};

} // namespace rund_node_test_virtual
