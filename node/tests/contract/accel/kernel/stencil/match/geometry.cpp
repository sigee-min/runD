#include "run.hpp"

#include <array>

namespace node_accel_contract::stencil {

bool MatchesWideWindowU32(const rund::AccelDevice &pick) {
  return MatchesReference<rund::kernel::u32>(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::U32, rund::kernel::StencilOp::Sum,
      rund::kernel::StencilElement::U32, 2u,
      std::array<rund::kernel::u32, 6u>{1u, 4u, 2u, 8u, 16u, 32u});
}

bool MatchesCount65U32(const rund::AccelDevice &pick) {
  std::array<rund::kernel::u32, 65u> input{};
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] = static_cast<rund::kernel::u32>(index * 2246822519u +
                                                  (index % 7u) * 3266489917u);
  }
  return MatchesReference<rund::kernel::u32>(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::U32, rund::kernel::StencilOp::Sum,
      rund::kernel::StencilElement::U32, 1u, input);
}

bool MatchesRadius64BoundaryU32(const rund::AccelDevice &pick) {
  std::array<rund::kernel::u32, 259u> input{};
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] = static_cast<rund::kernel::u32>(index * 2654435761u +
                                                  (index % 11u) * 17u);
  }
  return MatchesReference<rund::kernel::u32>(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::U32, rund::kernel::StencilOp::Sum,
      rund::kernel::StencilElement::U32, 64u, input);
}

bool MatchesCapabilitySharedBoundaryU32(const rund::AccelDevice &pick,
                                        const rund::kernel::u64 radius) {
  std::array<rund::kernel::u32, 515u> input{};
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] = static_cast<rund::kernel::u32>(index * 2654435761u +
                                                  (index % 13u) * 2246822519u);
  }
  return MatchesForcedPathReference<rund::kernel::u32>(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::U32, rund::kernel::StencilOp::Sum,
      rund::kernel::StencilElement::U32, radius, input,
      match_detail::ForcedRangePath::SharedHalo);
}

} // namespace node_accel_contract::stencil
