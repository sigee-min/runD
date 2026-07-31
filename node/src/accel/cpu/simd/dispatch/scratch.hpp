#pragma once

#include "../context.hpp"

#include <array>

namespace rund::node::accel::cpu_simd_detail {

using CpuSimdScratchBytesFn = std::size_t (*)(
    const PreparedRun& prepared) noexcept;

[[nodiscard]] inline CpuSimdScratchBytesFn SelectCpuSimdScratchSizer(
    const rund::kernel::ComputeScalar scalar) noexcept {
  using rund::kernel::ComputeScalar;
  static constexpr std::array<CpuSimdScratchBytesFn, 3u> kSizerByScalar{
      nullptr, ScratchBytesFixedLane32, ScratchBytesFixedLane64};
  const auto slot = static_cast<std::size_t>(scalar);
  return slot < kSizerByScalar.size() ? kSizerByScalar[slot] : nullptr;
}

}  // namespace rund::node::accel::cpu_simd_detail
