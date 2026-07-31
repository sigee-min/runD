#pragma once

#include "run/execute.hpp"

#include <array>
#include <cstddef>

namespace node_accel_contract::collective {
namespace radix_stability {

inline constexpr std::size_t BlockSize = 256u;
inline constexpr std::size_t HalfSize = 1024u;
inline constexpr std::size_t ElementCount = HalfSize * 2u;
inline constexpr std::size_t AllEqualCount = HalfSize + 1u;

[[nodiscard]] constexpr std::array<rund::kernel::u32, ElementCount>
Fixture() noexcept {
  std::array<rund::kernel::u32, ElementCount> keys{};
  for (std::size_t lane = 0u; lane < HalfSize; ++lane) {
    keys[lane] = static_cast<rund::kernel::u32>(0x80u + lane % 0x80u);
    keys[HalfSize + lane] =
        static_cast<rund::kernel::u32>(0x100u | (lane % 0x80u));
  }
  return keys;
}

[[nodiscard]] constexpr std::array<rund::kernel::u32, AllEqualCount>
AllEqualFixture() noexcept {
  std::array<rund::kernel::u32, AllEqualCount> keys{};
  keys.fill(0x80808080u);
  return keys;
}

[[nodiscard]] constexpr bool CrossesOriginalBlocks() noexcept {
  constexpr auto keys = Fixture();
  for (std::size_t lane = 0u; lane < HalfSize; ++lane) {
    if ((keys[lane] & 0xffu) <= (keys[HalfSize + lane] & 0xffu) ||
        (keys[lane] & 0xff00u) == (keys[HalfSize + lane] & 0xff00u)) {
      return false;
    }
  }
  return true;
}

static_assert(HalfSize % BlockSize == 0u);
static_assert(ElementCount >= BlockSize * 2u);
static_assert(CrossesOriginalBlocks());

} // namespace radix_stability

[[nodiscard]] inline bool
RadixBlocksRemainStable(const rund::AccelDevice &pick) {
  return SortMatchesCpuReference(pick, rund::kernel::ComputeScalar::Lane32,
                                 radix_stability::Fixture(), 16u,
                                 rund::kernel::ComputeDomain::U32) &&
         SortMatchesCpuReference(pick, rund::kernel::ComputeScalar::Lane32,
                                 radix_stability::AllEqualFixture(), 32u,
                                 rund::kernel::ComputeDomain::U32);
}

} // namespace node_accel_contract::collective
