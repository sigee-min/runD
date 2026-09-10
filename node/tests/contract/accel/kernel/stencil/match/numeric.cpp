#include "run.hpp"

#include <array>
#include <limits>

namespace node_accel_contract::stencil {

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

bool MatchesBlockPrefixSuffixMinI32(const rund::AccelDevice &pick) {
  std::array<rund::kernel::u32, 257u> input{};
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] = static_cast<rund::kernel::u32>(index * 747796405u +
                                                  (index % 17u) * 2891336453u);
  }
  input[31u] = 0x80000000u;
  input[197u] = 0x7fffffffu;
  return MatchesReference<rund::kernel::u32>(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::I32, rund::kernel::StencilOp::Min,
      rund::kernel::StencilElement::U32, 257u, input);
}

bool MatchesBlockPrefixSuffixMaxI32(const rund::AccelDevice &pick) {
  std::array<rund::kernel::u32, 257u> input{};
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] = static_cast<rund::kernel::u32>(index * 747796405u +
                                                  (index % 17u) * 2891336453u);
  }
  input[31u] = 0x80000000u;
  input[197u] = 0x7fffffffu;
  return MatchesReference<rund::kernel::u32>(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::I32, rund::kernel::StencilOp::Max,
      rund::kernel::StencilElement::U32, 257u, input);
}

bool MatchesBlockPrefixSuffixMinU64(const rund::AccelDevice &pick) {
  std::array<rund::kernel::u64, 257u> input{};
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] =
        static_cast<rund::kernel::u64>(index) * 11400714819323198485ull +
        static_cast<rund::kernel::u64>(index % 19u) * 14029467366897019727ull;
  }
  input[43u] = 0u;
  input[211u] = std::numeric_limits<rund::kernel::u64>::max();
  return MatchesReference<rund::kernel::u64>(
      pick, rund::kernel::ComputeScalar::Lane64,
      rund::kernel::ComputeDomain::U64, rund::kernel::StencilOp::Min,
      rund::kernel::StencilElement::U64, 257u, input);
}

bool MatchesBlockPrefixSuffixMaxU64(const rund::AccelDevice &pick) {
  std::array<rund::kernel::u64, 257u> input{};
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] =
        static_cast<rund::kernel::u64>(index) * 11400714819323198485ull +
        static_cast<rund::kernel::u64>(index % 19u) * 14029467366897019727ull;
  }
  input[43u] = 0u;
  input[211u] = std::numeric_limits<rund::kernel::u64>::max();
  return MatchesReference<rund::kernel::u64>(
      pick, rund::kernel::ComputeScalar::Lane64,
      rund::kernel::ComputeDomain::U64, rund::kernel::StencilOp::Max,
      rund::kernel::StencilElement::U64, 257u, input);
}

} // namespace node_accel_contract::stencil
