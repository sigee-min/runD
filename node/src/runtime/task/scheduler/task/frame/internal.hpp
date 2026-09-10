#pragma once

#include "../frame.hpp"

#include <array>
#include <atomic>
#include <mutex>
#include <vector>

namespace rund::node::frame_detail {

inline constexpr std::size_t FrameAlignment = 16u;
inline constexpr std::uint32_t CompactFrameBytes = 160u;
inline constexpr std::uint32_t FramePageSlots = 256u;
inline constexpr std::uint32_t TierBit = 1u << 31u;
inline constexpr std::uint32_t SlotMask = TierBit - 1u;

} // namespace rund::node::frame_detail

namespace rund::node {

struct FrameArena::Store final {
  struct Tier final {
    std::vector<void *> pages{};
    std::vector<void *> blocks{};
    std::size_t stride_bytes{};
    std::uint32_t frame_bytes{};
    std::vector<std::uint32_t> free{};
    std::vector<std::uint32_t> generation{};
    std::vector<std::uint32_t> bytes{};
    std::vector<std::uint8_t> live{};
  };

  ~Store();

  std::atomic<std::uint64_t> refs{1u};
  std::mutex mutex{};
  std::size_t prefix_bytes{};
  std::array<Tier, 2u> tiers{};
  std::uint8_t tier_count{};
  FrameLimits limits{};
  FrameStats stats{};
  ReasonCode code{ReasonCode::Ok};
  bool retired{};
};

struct FrameArena::Header final {
  Store *store{};
  std::uint32_t slot_and_tier{};
  std::uint32_t generation{};
};

} // namespace rund::node
