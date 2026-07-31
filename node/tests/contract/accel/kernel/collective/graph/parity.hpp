#pragma once

#include <accel/device.hpp>

#include "../../compact/local.hpp"
#include "../../scan/local.hpp"
#include "../../sort/identity.hpp"
#include "../sort.hpp"
#include "scan.hpp"

#include <iostream>

namespace node_accel_contract::collective::graph_case {

[[nodiscard]] inline bool
RepresentativeCollectivesMatchCpu(const rund::AccelDevice &pick) {
  const auto require = [](const bool ok, const char *const name) {
    if (!ok) {
      std::cerr << "collective parity failure: " << name << '\n';
    }
    return ok;
  };
  return require(ScanMatchesCpuReference<rund::kernel::u32>(
                     pick, rund::kernel::ScanElement::U32,
                     rund::kernel::ComputeScalar::Lane32,
                     std::array<rund::kernel::u32, 8u>{3u, 1u, 4u, 0u, 2u, 5u,
                                                       1u, 6u}),
                 "scan-u32") &&
         require(ScanMatchesCpuReference<rund::kernel::u64>(
                     pick, rund::kernel::ScanElement::U64,
                     rund::kernel::ComputeScalar::Lane64,
                     std::array<rund::kernel::u64, 8u>{9u, 2u, 6u, 5u, 3u, 5u,
                                                       8u, 9u}),
                 "scan-u64") &&
         require(ScanThenMapPreservesInternalRoundtrip(pick),
                 "scan-map-roundtrip") &&
         require(
             SortMatchesCpuReference<rund::kernel::u32>(
                 pick, rund::kernel::ComputeScalar::Lane32, SortU32Fixture()),
             "sort-u32") &&
         require(SortMatchesCpuReference<rund::kernel::u32>(
                     pick, rund::kernel::ComputeScalar::Lane32,
                     SortU32Fixture(), 16u),
                 "sort-u32-key16") &&
         require(SortIdentityU32MatchesCpuReference(
                     pick, rund::kernel::ComputeScalar::Lane32,
                     SortU32Fixture(), 16u),
                 "sort-identity-u32-key16") &&
         require(
             SortMatchesCpuReference<rund::kernel::u64>(
                 pick, rund::kernel::ComputeScalar::Lane64, SortU64Fixture()),
             "sort-u64") &&
         require(CompactMatchesCpuReference(pick,
                                            rund::kernel::ComputeScalar::Lane32,
                                            CompactFixtureA(), 8u),
                 "compact") &&
         require(CompactRejectsCapacityInsufficient(
                     pick, rund::kernel::ComputeScalar::Lane32),
                 "compact-capacity-rejection");
}

} // namespace node_accel_contract::collective::graph_case
