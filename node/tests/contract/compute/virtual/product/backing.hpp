#pragma once

#include "model.hpp"

#include <rund/compute/virtual.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace rund_node_test_virtual::product {

// Public-product backing fixture. size_bytes() exposes only logical bytes;
// rounded guard storage remains poisoned so a partial terminal page cannot be
// silently written as a full page.
class MemoryVirtualBacking final : public rund::compute::VirtualBacking {
public:
  MemoryVirtualBacking(std::size_t logical_bytes, std::size_t page_bytes);

  [[nodiscard]] std::uint64_t size_bytes() const noexcept override {
    return logical_bytes_;
  }
  [[nodiscard]] rund::compute::Status
  read(std::uint64_t offset, std::span<std::byte> output) noexcept override;
  [[nodiscard]] rund::compute::Status
  write(std::uint64_t offset,
        std::span<const std::byte> input) noexcept override;

  [[nodiscard]] bool seed(std::span<const std::byte> input) noexcept;
  [[nodiscard]] bool observe(std::span<std::byte> output) noexcept;
  void reset(std::byte value) noexcept;
  void fail_next_read(rund::compute::Reason reason) noexcept;
  void fail_next_write(rund::compute::Reason reason) noexcept;
  void fail_next_write_after(std::size_t prefix_bytes,
                             rund::compute::Reason reason) noexcept;
  [[nodiscard]] bool tail_poisoned() const noexcept;
  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] const void *identity() const noexcept { return bytes_.data(); }
  [[nodiscard]] BackingFacts facts() const noexcept { return facts_; }

private:
  [[nodiscard]] bool contains(std::uint64_t offset,
                              std::size_t bytes) const noexcept;

  std::vector<std::byte> bytes_;
  std::size_t logical_bytes_{};
  std::size_t page_bytes_{};
  BackingFacts facts_{};
  rund::compute::Reason read_failure_{rund::compute::Reason::Ok};
  rund::compute::Reason write_failure_{rund::compute::Reason::Ok};
  std::size_t write_failure_prefix_{};
};

} // namespace rund_node_test_virtual::product
