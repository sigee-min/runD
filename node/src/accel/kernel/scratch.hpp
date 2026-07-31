#pragma once

#include "bindings/refs.hpp"

#include <accel/context/value.hpp>
#include <accel/kernel/value.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace rund::node::accel::detail {

struct KernelScratchPlan final {
  std::uint64_t last_bytes{};
  std::size_t page_count{};
  bool ok{};
  const char *reason{"accel_kernel_scratch_invalid"};
};

struct KernelScratchPage final {
  std::size_t slot{};
  std::uint64_t bytes{};
};

using KernelScratchLayout = std::vector<KernelScratchPage>;

namespace scratch {

struct Placement final {
  std::size_t page{};
  std::uint64_t offset{};
  bool ok{};
};

[[nodiscard]] inline bool align(const std::uint64_t value,
                                const std::uint64_t alignment,
                                std::uint64_t &result) noexcept {
  if (alignment == 0u || (alignment & (alignment - 1u)) != 0u) {
    return false;
  }
  const std::uint64_t mask = alignment - 1u;
  if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
    return false;
  }
  result = (value + mask) & ~mask;
  return true;
}

template <class Pages>
[[nodiscard]] Placement fit(Pages &pages, const std::uint64_t alignment,
                            const std::uint64_t bytes) noexcept {
  if (bytes == 0u) {
    return {};
  }
  for (std::size_t index = 0u; index < pages.size(); ++index) {
    auto &page = pages[index];
    std::uint64_t offset = 0u;
    if (!align(page.used, alignment, offset) || offset > page.bytes ||
        bytes > page.bytes - offset) {
      continue;
    }
    page.used = offset + bytes;
    return Placement{.page = index, .offset = offset, .ok = true};
  }
  return {};
}

template <class Pages> void reset(Pages &pages) noexcept {
  for (auto &page : pages) {
    page.used = 0u;
  }
}

template <class Pages>
[[nodiscard]] bool active(const Pages &pages) noexcept {
  for (const auto &page : pages) {
    if (page.used != 0u) {
      return true;
    }
  }
  return false;
}

} // namespace scratch

[[nodiscard]] KernelScratchPlan
PlanKernelScratch(const rund::AccelContext &context,
                  const rund::AccelKernel &kernel,
                  std::uint64_t alignment, std::uint64_t page_bytes);

[[nodiscard]] bool ValidKernelScratch(const KernelScratchLayout &layout,
                                      const RunBinds &binds) noexcept;

} // namespace rund::node::accel::detail
