#include "run.hpp"

#include <array>

namespace node_accel_contract::stencil {

bool MatchesPrefixDifferenceU32(const rund::AccelDevice &pick) {
  std::array<rund::kernel::u32, 257u> input{};
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] =
        static_cast<rund::kernel::u32>(index * index + 3u * index + 1u);
  }
  return MatchesReference<rund::kernel::u32>(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::U32, rund::kernel::StencilOp::Sum,
      rund::kernel::StencilElement::U32, 257u, input);
}

bool MatchesForcedPrefixDifferenceU32(const rund::AccelDevice &pick) {
  std::array<rund::kernel::u32, 257u> input{};
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] =
        static_cast<rund::kernel::u32>(index * index + 3u * index + 1u);
  }
  return MatchesDirectAndForcedRangePath(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::U32, rund::kernel::StencilOp::Sum,
      rund::kernel::StencilElement::U32, 257u, input,
      match_detail::ForcedRangePath::PrefixDifference);
}

bool MatchesForcedPrefixDifferenceU64(const rund::AccelDevice &pick) {
  std::array<rund::kernel::u64, 257u> input{};
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] =
        static_cast<rund::kernel::u64>(index) * 11400714819323198485ull +
        static_cast<rund::kernel::u64>(index % 23u) * 14029467366897019727ull;
  }
  return MatchesDirectAndForcedRangePath(
      pick, rund::kernel::ComputeScalar::Lane64,
      rund::kernel::ComputeDomain::U64, rund::kernel::StencilOp::Sum,
      rund::kernel::StencilElement::U64, 257u, input,
      match_detail::ForcedRangePath::PrefixDifference);
}

// 65,537 is deliberately one past 256^2.  It creates at least three prefix
// hierarchy levels for every legal Range width (64, 128, or 256),
// so the run proves that summary-role bindings and reverse fix-up are not
// limited to the first local block layer.
bool MatchesDeepPrefixHierarchy(const rund::AccelDevice &pick) {
  std::array<rund::kernel::u32, 65537u> input{};
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] = static_cast<rund::kernel::u32>(index * 2654435761u +
                                                  (index % 29u) * 2246822519u);
  }
  if (!MatchesForcedPathReference(
          pick, rund::kernel::ComputeScalar::Lane32,
          rund::kernel::ComputeDomain::U32, rund::kernel::StencilOp::Sum,
          rund::kernel::StencilElement::U32, 257u, input,
          match_detail::ForcedRangePath::PrefixDifference)) {
    return false;
  }
  std::array<rund::kernel::u64, 65537u> wide{};
  for (std::size_t index = 0u; index < wide.size(); ++index) {
    wide[index] =
        static_cast<rund::kernel::u64>(index) * 11400714819323198485ull +
        0xffffffffffffffffull;
  }
  return MatchesForcedPathReference(
      pick, rund::kernel::ComputeScalar::Lane64,
      rund::kernel::ComputeDomain::U64, rund::kernel::StencilOp::Sum,
      rund::kernel::StencilElement::U64, 257u, wide,
      match_detail::ForcedRangePath::PrefixDifference);
}

namespace {
template <typename T, std::size_t Count>
bool TiledTail(const rund::AccelDevice &pick) {
  std::array<T, Count> input{};
  for (std::size_t i = 0; i < Count; ++i) {
    input[i] = static_cast<T>(i) * static_cast<T>(11400714819323198485ull) +
               static_cast<T>(-1);
  }
  constexpr bool wide = sizeof(T) == 8u;
  return MatchesForcedPathReference(
      pick,
      wide ? rund::kernel::ComputeScalar::Lane64
           : rund::kernel::ComputeScalar::Lane32,
      wide ? rund::kernel::ComputeDomain::U64
           : rund::kernel::ComputeDomain::U32,
      rund::kernel::StencilOp::Sum,
      wide ? rund::kernel::StencilElement::U64
           : rund::kernel::StencilElement::U32,
      (Count < 257u ? Count : 257u), input,
      match_detail::ForcedRangePath::TiledDifference);
}
} // namespace
bool MatchesTiledDifference(const rund::AccelDevice &pick) {
  return TiledTail<rund::kernel::u32, 257u>(pick) &&
         TiledTail<rund::kernel::u32, 31u>(pick) &&
         TiledTail<rund::kernel::u32, 4095u>(pick) &&
         TiledTail<rund::kernel::u32, 4096u>(pick) &&
         TiledTail<rund::kernel::u32, 4097u>(pick) &&
         TiledTail<rund::kernel::u64, 257u>(pick) &&
         TiledTail<rund::kernel::u64, 31u>(pick) &&
         TiledTail<rund::kernel::u64, 4095u>(pick) &&
         TiledTail<rund::kernel::u64, 4096u>(pick) &&
         TiledTail<rund::kernel::u64, 4097u>(pick);
}

} // namespace node_accel_contract::stencil
