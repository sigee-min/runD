#pragma once

#include "../../memory/arena.hpp"
#include "../memory.hpp"

#include <kernel/core/checked.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace rund::compute::detail::resource_detail::memory_detail {

inline constexpr std::uint64_t Alignment = memory::Alignment;
inline constexpr std::uint64_t Chunk = memory::Chunk;

struct Lifetime final {
  std::uint32_t first{resource::NoNode};
  std::uint32_t last{resource::NoNode};
  std::uint32_t first_read{resource::NoNode};
  std::uint32_t first_write{resource::NoNode};
  bool first_write_dense{};
  bool first_write_complete{};
  Domain domain{};
  bool first_write_domain{};
};

struct Layout final {
  std::uint64_t bytes{};
  std::uint64_t count{};
  std::vector<std::uint32_t> owners{};
  std::vector<std::uint64_t> offsets{};
};

[[nodiscard]] inline bool aligned(const std::uint64_t value,
                                  const std::uint64_t alignment,
                                  std::uint64_t &result) noexcept {
  std::uint64_t padded = 0u;
  if (alignment == 0u || (alignment & (alignment - 1u)) != 0u ||
      !kernel::checked::add(value, alignment - 1u, padded)) {
    return false;
  }
  result = padded & ~(alignment - 1u);
  return true;
}

[[nodiscard]] bool pack(std::span<const graph::Resource> resources,
                        std::span<const Lifetime> lifetimes,
                        std::span<const std::uint32_t> ids,
                        std::uint64_t limit_bytes, bool destructive,
                        Layout &result);

} // namespace rund::compute::detail::resource_detail::memory_detail
