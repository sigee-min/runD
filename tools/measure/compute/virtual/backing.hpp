#pragma once

#include <rund/compute/virtual.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace rund::measure::compute::virtual_residency {

class MemoryBacking final : public ::rund::compute::VirtualBacking {
public:
  explicit MemoryBacking(std::size_t bytes);

  [[nodiscard]] std::uint64_t size_bytes() const noexcept override;
  [[nodiscard]] ::rund::compute::Status
  read(std::uint64_t offset, std::span<std::byte> output) noexcept override;
  [[nodiscard]] ::rund::compute::Status
  write(std::uint64_t offset,
        std::span<const std::byte> input) noexcept override;
  void reset(std::byte value) noexcept;

private:
  [[nodiscard]] bool contains(std::uint64_t offset,
                              std::size_t bytes) const noexcept;

  std::vector<std::byte> bytes_;
};

} // namespace rund::measure::compute::virtual_residency
