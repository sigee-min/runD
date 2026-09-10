#pragma once

#include <accel/device.hpp>

#include <bit>
#include <limits>
#include <span>
#include <vector>

namespace node_accel_contract::reduce {

bool MatchesU32(const rund::AccelDevice &pick) {
  return MatchesReference<rund::kernel::u32>(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::U32, rund::kernel::ReduceOp::Sum,
      rund::kernel::ReduceElement::U32,
      std::array<rund::kernel::u32, 9u>{1u, 3u, 5u, 7u, 9u, 11u, 13u, 15u,
                                        17u});
}

bool MatchesU64(const rund::AccelDevice &pick) {
  return MatchesReference<rund::kernel::u64>(
      pick, rund::kernel::ComputeScalar::Lane64,
      rund::kernel::ComputeDomain::U64, rund::kernel::ReduceOp::Sum,
      rund::kernel::ReduceElement::U64,
      std::array<rund::kernel::u64, 9u>{2u, 4u, 6u, 8u, 10u, 12u, 14u, 16u,
                                        18u});
}

bool CountsNonzeroU32(const rund::AccelDevice &pick) {
  return MatchesReference<rund::kernel::u32>(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::U32, rund::kernel::ReduceOp::CountNonzero,
      rund::kernel::ReduceElement::U32,
      std::array<rund::kernel::u32, 9u>{0u, 0xffffffffu, 0u, 0x80000000u,
                                        0x40000001u, 0u, 0x7fffffffu, 0u,
                                        0xf0000003u});
}

bool CountsNonzeroU64(const rund::AccelDevice &pick) {
  return MatchesReference<rund::kernel::u64>(
      pick, rund::kernel::ComputeScalar::Lane64,
      rund::kernel::ComputeDomain::U64, rund::kernel::ReduceOp::CountNonzero,
      rund::kernel::ReduceElement::U64,
      std::array<rund::kernel::u64, 9u>{
          0u, 0xffffffffffffffffull, 0u, 0x8000000000000000ull,
          0x0000000100000000ull, 0u, 0x00000000ffffffffull, 0u,
          0x7000000000000003ull});
}

bool MatchesWideHierarchyU32(const rund::AccelDevice &pick) {
  const std::vector<rund::kernel::u32> input(262144u, 1u);
  return MatchesReference<rund::kernel::u32>(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::U32, rund::kernel::ReduceOp::Sum,
      rund::kernel::ReduceElement::U32, input, 256u);
}

bool MatchesWideHierarchyI64Cancellation(const rund::AccelDevice &pick) {
  constexpr rund::kernel::i64 high =
      std::numeric_limits<rund::kernel::i64>::max();
  std::vector<rund::kernel::u64> input(4097u);
  for (std::size_t index = 0u; index + 1u < input.size(); index += 2u) {
    input[index] = std::bit_cast<rund::kernel::u64>(high);
    input[index + 1u] = std::bit_cast<rund::kernel::u64>(-high);
  }
  input.back() = 7u;
  return MatchesReference<rund::kernel::u64>(
      pick, rund::kernel::ComputeScalar::Lane64,
      rund::kernel::ComputeDomain::I64, rund::kernel::ReduceOp::Sum,
      rund::kernel::ReduceElement::U64, input, 256u);
}

template <typename T>
[[nodiscard]] bool
RejectsWideOverflow(const rund::AccelDevice &pick,
                    const rund::kernel::ComputeScalar scalar,
                    const rund::kernel::ComputeDomain domain,
                    const rund::kernel::ReduceElement element) {
  std::vector<T> input(4097u, 0u);
  input.front() = std::numeric_limits<T>::max();
  input.back() = 1u;
  match::Resources<T> resources =
      match::BuildResources(pick, scalar, domain, rund::kernel::ReduceOp::Sum,
                            element, std::span<const T>{input}, 256u);
  if (!resources.kernel.check.ok) {
    return false;
  }
  const auto bindings = match::Bindings(resources);
  const rund::AccelEvidence evidence =
      rund::node::accel::RunAccelKernel(resources.context, resources.kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = input.size(),
                                            .fresh_evidence = true,
                                        });
  return primitive::EvidenceReason(evidence, "compute_reduce_sum_overflow") &&
         evidence.run.transfer.host_to_device_bytes == 0u &&
         evidence.run.transfer.device_to_host_bytes == 0u;
}

bool RejectsWideU32Overflow(const rund::AccelDevice &pick) {
  return RejectsWideOverflow<rund::kernel::u32>(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::U32, rund::kernel::ReduceElement::U32);
}

bool RejectsWideU64Overflow(const rund::AccelDevice &pick) {
  return RejectsWideOverflow<rund::kernel::u64>(
      pick, rund::kernel::ComputeScalar::Lane64,
      rund::kernel::ComputeDomain::U64, rund::kernel::ReduceElement::U64);
}

bool MatchesMinU32(const rund::AccelDevice &pick) {
  return MatchesReference<rund::kernel::u32>(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::U32, rund::kernel::ReduceOp::Min,
      rund::kernel::ReduceElement::U32,
      std::array<rund::kernel::u32, 9u>{19u, 5u, 17u, 3u, 23u, 11u, 7u, 13u,
                                        29u});
}

bool MatchesMaxU64(const rund::AccelDevice &pick) {
  return MatchesReference<rund::kernel::u64>(
      pick, rund::kernel::ComputeScalar::Lane64,
      rund::kernel::ComputeDomain::U64, rund::kernel::ReduceOp::Max,
      rund::kernel::ReduceElement::U64,
      std::array<rund::kernel::u64, 9u>{20u, 4u, 60u, 8u, 10u, 100u, 14u, 0u,
                                        18u});
}

bool MatchesMinU32NonPowerOfTwoBlock(const rund::AccelDevice &pick) {
  return MatchesReference<rund::kernel::u32>(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::U32, rund::kernel::ReduceOp::Min,
      rund::kernel::ReduceElement::U32,
      std::array<rund::kernel::u32, 9u>{19u, 17u, 1u, 23u, 11u, 7u, 13u, 29u,
                                        31u},
      3u);
}

bool MatchesMaxU64NonPowerOfTwoBlock(const rund::AccelDevice &pick) {
  return MatchesReference<rund::kernel::u64>(
      pick, rund::kernel::ComputeScalar::Lane64,
      rund::kernel::ComputeDomain::U64, rund::kernel::ReduceOp::Max,
      rund::kernel::ReduceElement::U64,
      std::array<rund::kernel::u64, 9u>{20u, 4u, 100u, 8u, 10u, 60u, 14u, 0u,
                                        18u},
      3u);
}

bool MatchesMinI32(const rund::AccelDevice &pick) {
  return MatchesReference<rund::kernel::u32>(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::I32, rund::kernel::ReduceOp::Min,
      rund::kernel::ReduceElement::U32,
      std::array<rund::kernel::u32, 9u>{0xffffffffu, 5u, 0xfffffffdu, 3u, 23u,
                                        11u, 7u, 13u, 0xfffffff9u});
}

template <typename T>
bool MatchesExtremaSimdBoundaries(const rund::AccelDevice &pick) {
  constexpr bool wide = sizeof(T) == 8u;
  const auto scalar = wide ? rund::kernel::ComputeScalar::Lane64
                           : rund::kernel::ComputeScalar::Lane32;
  const auto element = wide ? rund::kernel::ReduceElement::U64
                            : rund::kernel::ReduceElement::U32;
  for (const auto block : {3u, 33u, 65u, 256u}) {
    // Partial SIMD groups and a partial final workgroup, followed by at least
    // one more pass and multiple packed physical groups. High-word ties
    // exercise the U64 low-word winner mask.
    std::vector<T> input(67u * block + 3u);
    for (std::size_t i = 0u; i < input.size(); ++i) {
      input[i] =
          static_cast<T>((static_cast<rund::kernel::u64>(i % 3u) << 32u) |
                         static_cast<rund::kernel::u32>(i * 2654435761u));
      if ((i & 1u) != 0u) {
        input[i] ^= T{1u} << (sizeof(T) * 8u - 1u);
      }
    }
    for (const bool signed_domain : {false, true}) {
      const auto domain =
          wide ? (signed_domain ? rund::kernel::ComputeDomain::I64
                                : rund::kernel::ComputeDomain::U64)
               : (signed_domain ? rund::kernel::ComputeDomain::I32
                                : rund::kernel::ComputeDomain::U32);
      for (const auto op :
           {rund::kernel::ReduceOp::Min, rund::kernel::ReduceOp::Max}) {
        if (!MatchesReference<T>(pick, scalar, domain, op, element, input,
                                 block)) {
          return false;
        }
      }
    }
  }
  return true;
}

} // namespace node_accel_contract::reduce
