#include "run.hpp"

#include <array>
#include <limits>

namespace node_accel_contract::stencil {

bool MatchesForcedBlockPrefixSuffixMinI32(const rund::AccelDevice &pick) {
  std::array<rund::kernel::u32, 257u> input{};
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] = static_cast<rund::kernel::u32>(index * 747796405u +
                                                  (index % 17u) * 2891336453u);
  }
  input[31u] = 0x80000000u;
  input[197u] = 0x7fffffffu;
  return MatchesDirectAndForcedRangePath(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::I32, rund::kernel::StencilOp::Min,
      rund::kernel::StencilElement::U32, 257u, input,
      match_detail::ForcedRangePath::BlockPrefixSuffix);
}

bool MatchesForcedBlockPrefixSuffixMaxI32(const rund::AccelDevice &pick) {
  std::array<rund::kernel::u32, 257u> input{};
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] = static_cast<rund::kernel::u32>(index * 747796405u +
                                                  (index % 17u) * 2891336453u);
  }
  input[31u] = 0x80000000u;
  input[197u] = 0x7fffffffu;
  return MatchesDirectAndForcedRangePath(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::I32, rund::kernel::StencilOp::Max,
      rund::kernel::StencilElement::U32, 257u, input,
      match_detail::ForcedRangePath::BlockPrefixSuffix);
}

bool MatchesForcedBlockPrefixSuffixMinU64(const rund::AccelDevice &pick) {
  std::array<rund::kernel::u64, 257u> input{};
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] =
        static_cast<rund::kernel::u64>(index) * 11400714819323198485ull +
        static_cast<rund::kernel::u64>(index % 19u) * 14029467366897019727ull;
  }
  input[43u] = 0u;
  input[211u] = std::numeric_limits<rund::kernel::u64>::max();
  return MatchesDirectAndForcedRangePath(
      pick, rund::kernel::ComputeScalar::Lane64,
      rund::kernel::ComputeDomain::U64, rund::kernel::StencilOp::Min,
      rund::kernel::StencilElement::U64, 257u, input,
      match_detail::ForcedRangePath::BlockPrefixSuffix);
}

bool MatchesForcedBlockPrefixSuffixMaxU64(const rund::AccelDevice &pick) {
  std::array<rund::kernel::u64, 257u> input{};
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] =
        static_cast<rund::kernel::u64>(index) * 11400714819323198485ull +
        static_cast<rund::kernel::u64>(index % 19u) * 14029467366897019727ull;
  }
  input[43u] = 0u;
  input[211u] = std::numeric_limits<rund::kernel::u64>::max();
  return MatchesDirectAndForcedRangePath(
      pick, rund::kernel::ComputeScalar::Lane64,
      rund::kernel::ComputeDomain::U64, rund::kernel::StencilOp::Max,
      rund::kernel::StencilElement::U64, 257u, input,
      match_detail::ForcedRangePath::BlockPrefixSuffix);
}

} // namespace node_accel_contract::stencil
