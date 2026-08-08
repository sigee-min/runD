#pragma once

#include <accel/device.hpp>

#include "execute.hpp"

#include <limits>

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
  return MatchesReference<rund::kernel::u32>(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::U32, rund::kernel::StencilOp::Sum,
      rund::kernel::StencilElement::U32, radius, input);
}

bool MatchesDirectVariantU32(const rund::AccelDevice &pick) {
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

bool MatchesDirectMinI32(const rund::AccelDevice &pick) {
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

bool MatchesDirectMaxI32(const rund::AccelDevice &pick) {
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

bool MatchesDirectMinU64(const rund::AccelDevice &pick) {
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

bool MatchesDirectMaxU64(const rund::AccelDevice &pick) {
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
