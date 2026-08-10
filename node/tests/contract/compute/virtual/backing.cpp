#include "backing.hpp"

#include <algorithm>
#include <cstring>

namespace rund_node_test_virtual {

MemoryVirtualBacking::MemoryVirtualBacking(const std::size_t bytes,
                                           const std::byte initial)
    : bytes_(bytes, initial) {}

bool MemoryVirtualBacking::contains(const std::size_t offset,
                                    const std::size_t bytes) const noexcept {
  return offset <= bytes_.size() && bytes <= bytes_.size() - offset;
}

bool MemoryVirtualBacking::seed(
    const std::size_t offset, const std::span<const std::byte> input) noexcept {
  if (!contains(offset, input.size())) {
    return false;
  }
  if (!input.empty()) {
    std::memcpy(bytes_.data() + offset, input.data(), input.size());
  }
  return true;
}

bool MemoryVirtualBacking::load(const std::size_t offset,
                                const std::span<std::byte> output) noexcept {
  if (!contains(offset, output.size())) {
    return false;
  }
  if (!output.empty()) {
    std::memcpy(output.data(), bytes_.data() + offset, output.size());
  }
  ++facts_.load_count;
  facts_.load_bytes += output.size();
  return true;
}

bool MemoryVirtualBacking::store(
    const std::size_t offset, const std::span<const std::byte> input) noexcept {
  if (!contains(offset, input.size())) {
    return false;
  }
  if (!input.empty()) {
    std::memcpy(bytes_.data() + offset, input.data(), input.size());
  }
  ++facts_.store_count;
  facts_.store_bytes += input.size();
  return true;
}

bool MemoryVirtualBacking::observe(const std::size_t offset,
                                   const std::span<std::byte> output) noexcept {
  if (!contains(offset, output.size())) {
    return false;
  }
  if (!output.empty()) {
    std::memcpy(output.data(), bytes_.data() + offset, output.size());
  }
  ++facts_.observation_count;
  facts_.observation_bytes += output.size();
  return true;
}

bool MemoryVirtualBacking::tail_is(const std::size_t offset,
                                   const std::byte expected) const noexcept {
  return offset <= bytes_.size() &&
         std::all_of(
             bytes_.begin() + static_cast<std::ptrdiff_t>(offset), bytes_.end(),
             [expected](const std::byte value) { return value == expected; });
}

} // namespace rund_node_test_virtual
