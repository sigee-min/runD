#pragma once

#include <accel/device.hpp>

#include "execute.hpp"

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

bool MatchesWideWindowU32(const rund::AccelDevice &pick) {
  return MatchesReference<rund::kernel::u32>(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::U32, rund::kernel::StencilOp::Sum,
      rund::kernel::StencilElement::U32, 2u,
      std::array<rund::kernel::u32, 6u>{1u, 4u, 2u, 8u, 16u, 32u});
}

bool MatchesMinU32(const rund::AccelDevice &pick) {
  return MatchesReference<rund::kernel::u32>(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::U32, rund::kernel::StencilOp::Min,
      rund::kernel::StencilElement::U32, 1u,
      std::array<rund::kernel::u32, 6u>{5u, 2u, 9u, 1u, 7u, 3u});
}

bool MatchesMaxU64(const rund::AccelDevice &pick) {
  return MatchesReference<rund::kernel::u64>(
      pick, rund::kernel::ComputeScalar::Lane64,
      rund::kernel::ComputeDomain::U64, rund::kernel::StencilOp::Max,
      rund::kernel::StencilElement::U64, 1u,
      std::array<rund::kernel::u64, 6u>{5u, 2u, 9u, 1u, 7u, 3u});
}

bool MatchesMinI32(const rund::AccelDevice &pick) {
  return MatchesReference<rund::kernel::u32>(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::I32, rund::kernel::StencilOp::Min,
      rund::kernel::StencilElement::U32, 1u,
      std::array<rund::kernel::u32, 6u>{0xffffffffu, 2u, 0xfffffffdu, 1u, 7u,
                                        0xfffffff9u});
}

} // namespace node_accel_contract::stencil
