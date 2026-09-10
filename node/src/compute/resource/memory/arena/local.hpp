#pragma once

#include "../model.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace rund::compute::detail::resource_detail::memory_detail {

struct Placement final {
  std::uint64_t bytes{};
};

struct Arena final {
  std::uint64_t bytes{};
  std::uint64_t count{};
  std::vector<std::uint32_t> owners{};
};

[[nodiscard]] bool place_ordinary(std::span<const graph::Resource> resources,
                                  std::span<const Lifetime> lifetimes,
                                  std::span<const std::uint32_t> ids,
                                  std::uint64_t page_bytes, bool destructive,
                                  std::span<std::uint64_t> offsets,
                                  Placement &result);

[[nodiscard]] bool measure_ordinary(const Placement &placement,
                                    std::span<const graph::Resource> resources,
                                    std::span<const std::uint32_t> ids,
                                    std::span<const std::uint64_t> offsets,
                                    std::uint64_t page_bytes, Arena &result);

[[nodiscard]] bool place_large(std::span<const graph::Resource> resources,
                               std::span<const Lifetime> lifetimes,
                               std::span<const std::uint32_t> ids,
                               bool destructive, Layout &result);

} // namespace rund::compute::detail::resource_detail::memory_detail
