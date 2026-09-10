#pragma once

#include "../backing.hpp"

namespace rund::compute::detail {

// The public authority remains VirtualBacking. This private final type only
// supplies exact Buffer-backed byte I/O; the resident capability itself lives
// in VirtualBackingState and can be minted only by the factory TU.
class ResidentVirtualBacking final : public VirtualBacking {
public:
  [[nodiscard]] std::uint64_t size_bytes() const noexcept override;
  [[nodiscard]] Status read(std::uint64_t offset,
                            std::span<std::byte> output) noexcept override;
  [[nodiscard]] Status
  read_batch(std::span<const VirtualRead> ranges) noexcept override;
  [[nodiscard]] Status
  write(std::uint64_t offset,
        std::span<const std::byte> input) noexcept override;
};

} // namespace rund::compute::detail
