#pragma once

#include "../cache_model.hpp"

#include <rund/compute/pipeline/capacity.hpp>
#include <rund/compute/pipeline/shape.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace rund::compute::detail::residency {

class Authority;

struct VirtualTransactionLease final {
  static constexpr std::size_t RegionCapacity = 4u;
  static constexpr std::size_t Capacity =
      2u * PipelineLeafCapacity + 2u * PipelineIterationCapacity;

  struct Row final {
    std::uint32_t frame{};
    CacheKey key{};
    DirtyExtent dirty{};
    FrameTier tier{FrameTier::Device};
    FrameRole role{FrameRole::Output};
    FrameState state{FrameState::Empty};
    std::uint64_t owner_generation{};
  };

  VirtualTransactionLease() noexcept = default;
  VirtualTransactionLease(const VirtualTransactionLease &) = delete;
  VirtualTransactionLease &operator=(const VirtualTransactionLease &) = delete;
  VirtualTransactionLease(VirtualTransactionLease &&) noexcept = default;
  VirtualTransactionLease &
  operator=(VirtualTransactionLease &&) noexcept = default;

  [[nodiscard]] explicit operator bool() const noexcept {
    return active && authority != nullptr && gate.owns_lock();
  }

  std::array<std::uint32_t, Capacity> frames{};
  std::size_t row_count{};
  Authority *authority{};
  std::unique_lock<std::mutex> gate{};
  bool active{};
};

} // namespace rund::compute::detail::residency
