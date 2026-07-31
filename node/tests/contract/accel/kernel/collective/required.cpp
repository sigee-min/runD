#include <accel/api.hpp>
#include <accel/device.hpp>

#include "../compact/local.hpp"
#include "../scan/local.hpp"
#include "local.hpp"
#include "scan/run.hpp"
#include "sort.hpp"
#include "sort/bounded.hpp"

#include <node/accel/pick.hpp>

#include <array>

namespace node_accel_contract::collective {
namespace {

[[nodiscard]] bool RequiredBoundedSort(const rund::AccelApi api) {
  const rund::AccelDevice pick = rund::node::accel::PickAccel(Policy(api));
  return pick.check.ok
             ? bounded_sort::Contract(pick) && RadixBlocksRemainStable(pick)
             : PickUnavailableReasonIsPrecise(pick, api);
}

} // namespace

bool RequiredMetalBoundedSort() {
  return RequiredBoundedSort(rund::AccelApi::Metal);
}

bool RequiredVulkanBoundedSort() {
  return RequiredBoundedSort(rund::AccelApi::Vulkan);
}

bool RequiredMetalScan(const rund::AccelDevice &pick) {
  return pick.check.ok && pick.api == rund::AccelApi::Metal &&
         ScanMatchesCpuReference<rund::kernel::u32>(
             pick, rund::kernel::ScanElement::U32,
             rund::kernel::ComputeScalar::Lane32,
             std::array<rund::kernel::u32, 8u>{3u, 1u, 4u, 0u, 2u, 5u, 1u,
                                               6u}) &&
         ScanMatchesCpuReference<rund::kernel::u64>(
             pick, rund::kernel::ScanElement::U64,
             rund::kernel::ComputeScalar::Lane64,
             std::array<rund::kernel::u64, 8u>{9u, 2u, 6u, 5u, 3u, 5u, 8u,
                                               9u}) &&
         ScanThenMapPreservesInternalRoundtrip(pick) &&
         SortMatchesCpuReference<rund::kernel::u32>(
             pick, rund::kernel::ComputeScalar::Lane32, SortU32Fixture()) &&
         SortMatchesCpuReference<rund::kernel::u64>(
             pick, rund::kernel::ComputeScalar::Lane64, SortU64Fixture()) &&
         SortMatchesCpuReference<rund::kernel::u32>(
             pick, rund::kernel::ComputeScalar::Lane32,
             SortU32Declared16Fixture(), 16u) &&
         CompactMatchesCpuReference(pick, rund::kernel::ComputeScalar::Lane32,
                                    CompactFixtureA(), 8u) &&
         CompactRejectsCapacityInsufficient(
             pick, rund::kernel::ComputeScalar::Lane32) &&
         SortRejectsRunBufferShapeMismatch<rund::kernel::u32>(
             pick, rund::kernel::ComputeScalar::Lane32);
}

bool RequiredVulkanScan(const rund::AccelDevice &pick) {
  return pick.check.ok && pick.api == rund::AccelApi::Vulkan &&
         ScanMatchesCpuReference<rund::kernel::u32>(
             pick, rund::kernel::ScanElement::U32,
             rund::kernel::ComputeScalar::Lane32,
             std::array<rund::kernel::u32, 8u>{2u, 7u, 1u, 8u, 2u, 8u, 1u,
                                               8u}) &&
         ScanMatchesCpuReference<rund::kernel::u64>(
             pick, rund::kernel::ScanElement::U64,
             rund::kernel::ComputeScalar::Lane64,
             std::array<rund::kernel::u64, 8u>{1u, 6u, 1u, 8u, 0u, 3u, 3u,
                                               9u}) &&
         ScanThenMapPreservesInternalRoundtrip(pick) &&
         SortMatchesCpuReference<rund::kernel::u32>(
             pick, rund::kernel::ComputeScalar::Lane32, SortU32Fixture()) &&
         SortMatchesCpuReference<rund::kernel::u64>(
             pick, rund::kernel::ComputeScalar::Lane64, SortU64Fixture()) &&
         CompactMatchesCpuReference(pick, rund::kernel::ComputeScalar::Lane32,
                                    CompactFixtureB(), 8u) &&
         CompactRejectsCapacityInsufficient(
             pick, rund::kernel::ComputeScalar::Lane32) &&
         SortRejectsRunBufferShapeMismatch<rund::kernel::u64>(
             pick, rund::kernel::ComputeScalar::Lane64);
}

} // namespace node_accel_contract::collective

namespace node_accel_contract {

bool RequiredMetalRunsBoundedSort() {
  return collective::RequiredMetalBoundedSort();
}

bool RequiredVulkanRunsBoundedSort() {
  return collective::RequiredVulkanBoundedSort();
}

} // namespace node_accel_contract
