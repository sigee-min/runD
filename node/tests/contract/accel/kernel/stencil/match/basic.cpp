#include "run.hpp"

#include <array>

namespace node_accel_contract::stencil {

bool MatchesU32(const rund::AccelDevice &pick) {
  return MatchesReference<rund::kernel::u32>(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::U32, rund::kernel::StencilOp::Sum,
      rund::kernel::StencilElement::U32, 1u,
      std::array<rund::kernel::u32, 6u>{1u, 4u, 2u, 8u, 16u, 32u});
}

bool MatchesSumI32(const rund::AccelDevice &pick) {
  return MatchesReference<rund::kernel::u32>(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::I32, rund::kernel::StencilOp::Sum,
      rund::kernel::StencilElement::U32, 1u,
      std::array<rund::kernel::u32, 6u>{0xffffffffu, 4u, 0xfffffffdu, 8u, 16u,
                                        32u});
}

bool MatchesU64(const rund::AccelDevice &pick) {
  return MatchesReference<rund::kernel::u64>(
      pick, rund::kernel::ComputeScalar::Lane64,
      rund::kernel::ComputeDomain::U64, rund::kernel::StencilOp::Sum,
      rund::kernel::StencilElement::U64, 1u,
      std::array<rund::kernel::u64, 6u>{3u, 5u, 9u, 17u, 33u, 65u});
}

} // namespace node_accel_contract::stencil
