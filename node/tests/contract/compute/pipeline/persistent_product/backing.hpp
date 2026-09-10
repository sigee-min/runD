#pragma once

#include <rund/compute/virtual.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace rund_node_test_persistent_product {

class PersistentProductBacking final : public rund::compute::VirtualBacking {
public:
  explicit PersistentProductBacking(std::size_t bytes);

  [[nodiscard]] std::uint64_t size_bytes() const noexcept override;
  [[nodiscard]] rund::compute::VirtualBackingTier
  tier() const noexcept override {
    return rund::compute::VirtualBackingTier::Persistent;
  }
  [[nodiscard]] std::uint32_t max_parallel_reads() const noexcept override {
    return 2u;
  }
  [[nodiscard]] rund::compute::Status
  read(std::uint64_t, std::span<std::byte>) noexcept override;
  [[nodiscard]] rund::compute::Status
      read_batch(std::span<const rund::compute::VirtualRead>) noexcept override;
  [[nodiscard]] rund::compute::Status
  write(std::uint64_t, std::span<const std::byte>) noexcept override;

  [[nodiscard]] bool seed(std::span<const std::byte>) noexcept;
  [[nodiscard]] bool observe(std::span<std::byte>) const noexcept;
  void fail_next_read() noexcept;
  [[nodiscard]] std::uint64_t read_batch_calls() const noexcept;
  [[nodiscard]] std::uint64_t read_batch_ranges() const noexcept;

private:
  std::vector<std::byte> bytes_;
  std::uint64_t read_batch_calls_{};
  std::uint64_t read_batch_ranges_{};
  bool fail_read_once_{};
};

} // namespace rund_node_test_persistent_product
