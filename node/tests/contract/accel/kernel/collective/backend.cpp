#include <accel/api.hpp>
#include <accel/device.hpp>

#include "../compact/local.hpp"
#include "../sort/identity.hpp"
#include "../sort/identity/run.hpp"
#include "local.hpp"
#include "scan/run.hpp"
#include "sort.hpp"
#include "sort/bounded.hpp"
#include "src/accel/gather/status.hpp"
#include "src/accel/metal/scan/limits.hpp"
#include "src/accel/metal/scan/source.hpp"
#include "src/accel/metal/sort/source.hpp"
#include "src/accel/scan/prefix.hpp"
#include "src/accel/source/hash.hpp"
#include "src/accel/scan/shape.hpp"
#include "src/accel/scatter/reduce/status.hpp"
#include "src/accel/segmented/status.hpp"
#include "src/accel/sort/block/metal.hpp"
#include "src/accel/sort/block/vulkan.hpp"
#include "src/accel/vulkan/collective/chunk.hpp"
#include "src/accel/vulkan/scan/source.hpp"
#include "src/accel/vulkan/sort/local/api.hpp"

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <string_view>

namespace node_accel_contract::collective {
namespace {

[[nodiscard]] std::size_t Occurrences(const std::string &text,
                                      const std::string_view needle) {
  std::size_t count = 0u;
  std::size_t offset = 0u;
  while ((offset = text.find(needle, offset)) != std::string::npos) {
    ++count;
    offset += needle.size();
  }
  return count;
}

[[nodiscard]] constexpr bool CollectiveChunkMathIsExact() noexcept {
  using rund::node::accel::detail::CeilGroups;
  using rund::node::accel::detail::kScanPrefixWorkgroupWidth;
  using rund::node::accel::detail::PlanScanPrefixExecution;
  using rund::node::accel::detail::ScanDispatches;
  using rund::node::accel::detail::ScanPrefixDispatches;
  using rund::node::accel::detail::ScanPrefixHasOffset;
  using rund::node::accel::detail::ScanPrefixPayloadBytes;
  using rund::node::accel::detail::ScanPrefixTotalsBytes;
  using rund::node::accel::detail::SortDispatches;
  constexpr rund::kernel::ScanPlan single{
      .op = rund::kernel::ScanOp::InclusiveSum,
      .element = rund::kernel::ScanElement::U32,
      .element_count = 256u,
      .element_bytes = 4u,
      .block_size = 256u,
      .block_count = 1u,
      .pass_count = 1u,
      .temp_bytes = 1024u,
      .ok = true,
      .reason = "ok",
  };
  constexpr rund::kernel::ScanPlan multiple{
      .op = rund::kernel::ScanOp::InclusiveSum,
      .element = rund::kernel::ScanElement::U64,
      .element_count = 513u,
      .element_bytes = 8u,
      .block_size = 256u,
      .block_count = 3u,
      .pass_count = 2u,
      .temp_bytes = 4104u,
      .ok = true,
      .reason = "ok",
  };
  constexpr rund::kernel::ScanPlan malformed{
      .op = rund::kernel::ScanOp::InclusiveSum,
      .element = rund::kernel::ScanElement::U32,
      .element_count = 513u,
      .element_bytes = 4u,
      .block_size = 256u,
      .block_count = 2u,
      .pass_count = 2u,
      .temp_bytes = 2052u,
      .ok = true,
      .reason = "ok",
  };
  constexpr rund::kernel::ScanPlan chunked{
      .op = rund::kernel::ScanOp::InclusiveSum,
      .element = rund::kernel::ScanElement::U32,
      .element_count = 65'536u,
      .element_bytes = 4u,
      .block_size = 1u,
      .block_count = 65'536u,
      .pass_count = 2u,
      .temp_bytes = 262'144u,
      .ok = true,
      .reason = "ok",
  };
  constexpr auto single_prefix = PlanScanPrefixExecution(single);
  constexpr auto multiple_prefix = PlanScanPrefixExecution(multiple);
  constexpr auto malformed_prefix = PlanScanPrefixExecution(malformed);
  constexpr auto chunked_prefix = PlanScanPrefixExecution(chunked);
  return CeilGroups(0u, 1u) == 0u && CeilGroups(1u, 0u) == 0u &&
         CeilGroups(65'535u, 65'535u) == 1u &&
         CeilGroups(65'536u, 65'535u) == 2u &&
         kScanPrefixWorkgroupWidth == 128u && single_prefix.ok() &&
         single_prefix.stage_count() == 1u &&
         single_prefix.stage(0u).groups == 1u &&
         single_prefix.temporary_count() == 1u &&
         single_prefix.temporary(0u).bytes == 4u && multiple_prefix.ok() &&
         multiple_prefix.stage_count() == 3u &&
         multiple_prefix.stage(0u).groups == 3u &&
         multiple_prefix.stage(1u).groups == 1u &&
         multiple_prefix.stage(2u).groups == 3u &&
         !ScanPrefixHasOffset(single_prefix) &&
         ScanPrefixHasOffset(multiple_prefix) &&
         ScanPrefixDispatches(single_prefix, 65'535u) == 1u &&
         ScanPrefixDispatches(multiple_prefix, 65'535u) == 3u &&
         chunked_prefix.ok() && chunked_prefix.stage_count() == 3u &&
         ScanPrefixDispatches(chunked_prefix, 65'535u) == 5u &&
         multiple_prefix.temporary(0u).bytes == 24u &&
         multiple_prefix.temporary(0u).last_stage == 2u &&
         ScanPrefixPayloadBytes(multiple).has_value() &&
         *ScanPrefixPayloadBytes(multiple) == 4104u &&
         ScanPrefixTotalsBytes(multiple).has_value() &&
         *ScanPrefixTotalsBytes(multiple) == 24u && !malformed_prefix.ok() &&
         !ScanPrefixPayloadBytes(malformed).has_value() &&
         !ScanPrefixTotalsBytes(malformed).has_value() &&
         ScanDispatches(2u, 1'024u, 65'535u) == 3u &&
         ScanDispatches(1u, 65'536u, 65'535u) == 2u &&
         ScanDispatches(2u, 65'536u, 65'535u) == 5u &&
         SortDispatches(8u, 65'536u, 65'535u) == 49u;
}

[[nodiscard]] constexpr bool StatusDecodersAreCanonical() noexcept {
  using rund::node::accel::detail::GatherStatus;
  using rund::node::accel::detail::ScatterReduceStatus;
  using rund::node::accel::detail::SegmentedScanStatus;
  return GatherStatus(0u).ok && !GatherStatus(1u).ok &&
         std::string_view{GatherStatus(1u).reason} ==
             "compute_bounded_count_invalid" &&
         !GatherStatus(2u).ok && !GatherStatus(3u).ok &&
         std::string_view{GatherStatus(3u).reason} ==
             "compute_gather_invalid" &&
         ScatterReduceStatus(0u).ok && !ScatterReduceStatus(1u).ok &&
         !ScatterReduceStatus(2u).ok && !ScatterReduceStatus(3u).ok &&
         std::string_view{ScatterReduceStatus(3u).reason} ==
             "compute_scatter_reduce_buffer_invalid" &&
         SegmentedScanStatus(0u).ok && !SegmentedScanStatus(1u).ok &&
         !SegmentedScanStatus(2u).ok && !SegmentedScanStatus(3u).ok &&
         std::string_view{SegmentedScanStatus(3u).reason} ==
             "compute_segmented_scan_invalid";
}

[[nodiscard]] bool ResidentShapeOverflowIsRejected() {
  constexpr auto maximum = std::numeric_limits<rund::kernel::u64>::max();
  constexpr rund::kernel::ScanPlan plan{
      .op = rund::kernel::ScanOp::InclusiveSum,
      .element = rund::kernel::ScanElement::U64,
      .element_count = maximum,
      .element_bytes = sizeof(rund::kernel::u64),
      .block_size = 256u,
      .block_count = maximum / 256u + 1u,
      .pass_count = 2u,
      .count_source = rund::kernel::ComputeCountSource::Descriptor,
      .ok = true,
      .reason = "ok",
  };
  constexpr rund::kernel::ResidentBufferRef input{
      .id = 1u,
      .bytes = maximum,
      .element_bytes = sizeof(rund::kernel::u64),
      .stride_bytes = sizeof(rund::kernel::u64),
      .count = maximum,
      .usage = rund::kernel::kResidentUsageRead,
  };
  constexpr rund::kernel::ResidentBufferRef output{
      .id = 2u,
      .bytes = maximum,
      .element_bytes = sizeof(rund::kernel::u64),
      .stride_bytes = sizeof(rund::kernel::u64),
      .count = maximum,
      .usage = rund::kernel::kResidentUsageWrite,
  };
  const std::shared_ptr<void> input_handle = std::make_shared<int>(1);
  const std::shared_ptr<void> output_handle = std::make_shared<int>(2);
  const rund::node::accel::detail::ScanBinds bindings{
      .input = &input,
      .input_handle = &input_handle,
      .output = &output,
      .output_handle = &output_handle,
  };
  return !rund::node::accel::detail::ScanResidentShapeOk(plan, bindings);
}

[[nodiscard]] bool ScanSourceIsCanonical() {
  const std::string metal = rund::node::accel::detail::MetalScanSource();
  // Stage ownership changes must not silently invalidate native source keys.
  if (metal.size() != 15960u ||
      rund::node::accel::detail::SourceHash(metal) != 10312601048820643852ull) {
    return false;
  }
  const std::string width_declaration =
      "constant uint kScanWidth = " +
      std::to_string(rund::node::accel::detail::kScanPrefixWorkgroupWidth) +
      "u";
  if (metal.find("atomic_fetch_or_explicit(status, 1u") == std::string::npos ||
      metal.find(width_declaration) == std::string::npos ||
      metal.find("(block_size + ulong(width) - 1ul)") == std::string::npos ||
      Occurrences(metal, "value = input[index]") != 4u ||
      Occurrences(metal, "device const uint* input [[buffer(8)]]") != 1u ||
      Occurrences(metal, "device const ulong* input [[buffer(8)]]") != 1u) {
    return false;
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const std::string block = rund::node::accel::detail::VulkanScanSource(
      rund::kernel::ScanElement::U32, rund::kernel::ComputeDomain::I32,
      rund::node::accel::detail::VulkanScanStage::Block, true);
  const std::string exclusive = rund::node::accel::detail::VulkanScanSource(
      rund::kernel::ScanElement::U32, rund::kernel::ComputeDomain::I32,
      rund::node::accel::detail::VulkanScanStage::Block, false);
  const std::string prefix = rund::node::accel::detail::VulkanScanSource(
      rund::kernel::ScanElement::U32, rund::kernel::ComputeDomain::I32,
      rund::node::accel::detail::VulkanScanStage::Prefix, true);
  const std::string offset = rund::node::accel::detail::VulkanScanSource(
      rund::kernel::ScanElement::U32, rund::kernel::ComputeDomain::I32,
      rund::node::accel::detail::VulkanScanStage::Offset, true);
  const std::string local_size =
      "layout(local_size_x = " +
      std::to_string(rund::node::accel::detail::kScanPrefixWorkgroupWidth) +
      ") in";
  return block.find(local_size) != std::string::npos &&
         block.find("layout(push_constant)") != std::string::npos &&
         block.find("uint64_t(gl_WorkGroupID.x)") != std::string::npos &&
         block.find("lane_begin") != std::string::npos &&
         block.find("lane_totals") != std::string::npos &&
         block.find("step <<= 1u") != std::string::npos &&
         block.find("atomicOr(status[0]") != std::string::npos &&
         offset.find("atomicOr(status[0]") != std::string::npos &&
         exclusive.find("output_values[uint(index)] = running") !=
             std::string::npos &&
         prefix.find(local_size) != std::string::npos &&
         prefix.find("chunk_totals[2][kScanWidth]") != std::string::npos &&
         prefix.find("const uint64_t chunk_size") != std::string::npos &&
         prefix.find("step <<= 1u") != std::string::npos &&
         offset.find(local_size) != std::string::npos &&
         offset.find("input_values[uint(index)]") != std::string::npos;
#else
  return true;
#endif
}

[[nodiscard]] bool SignedSortSourcesCarryDomainOrder() {
  static_assert(rund::node::accel::detail::kMetalSortThreadCount == 256u);
  static_assert(rund::node::accel::detail::kMetalSortRounds == 8u);
  static_assert(rund::node::accel::detail::kMetalSortSimdWidth == 32u);
  static_assert(rund::node::accel::detail::kMetalSortSimdGroupCount == 8u);
  static_assert(rund::node::accel::detail::kMetalSortPackedPairs == 4u);
  static_assert(rund::node::accel::detail::kMetalSortBlockSize == 2048u);
  static_assert(rund::node::accel::detail::kVulkanSortRankGroupSize == 8u);
  static_assert(rund::node::accel::detail::kVulkanSortRankGroupCount == 64u);
  static_assert(rund::node::accel::detail::kVulkanSortGroupsPerWord == 8u);
  static_assert(rund::node::accel::detail::kVulkanSortPackedWordCount == 8u);
  static_assert(rund::node::accel::detail::kVulkanSortSharedBytes ==
                10u * 1024u);
  const std::string metal = rund::node::accel::detail::MetalSortSource(
      rund::node::accel::detail::kMetalSortBlockSize);
  if (metal.find("signed_order") == std::string::npos ||
      metal.find("bucket ^= 128u") == std::string::npos ||
      metal.find("rund_compute_sort_dispatch") == std::string::npos ||
      metal.find("dispatch_args[0] = range.invalid ? 0u : uint(blocks)") ==
          std::string::npos ||
      metal.find("active_blocks") == std::string::npos ||
      Occurrences(metal, "simd_ballot(") != 8u ||
      metal.find("atomic_fetch_or_explicit") == std::string::npos ||
      metal.find("round * kSortThreadCount + tid") == std::string::npos ||
      metal.find("tid * params.block_count + block") == std::string::npos ||
      metal.find("cursors[current_bank + bucket] + group_prefix + "
                 "subgroup_rank") == std::string::npos ||
      metal.find("rund_compute_sort_base") == std::string::npos) {
    return false;
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const std::string vulkan = rund::node::accel::detail::VulkanSortSource(
      rund::kernel::SortKey::U32,
      rund::node::accel::detail::SortStage::Scatter);
  const std::string dispatch = rund::node::accel::detail::VulkanSortSource(
      rund::kernel::SortKey::U32,
      rund::node::accel::detail::SortStage::Dispatch);
  if (vulkan.find("signed_order") == std::string::npos ||
      vulkan.find("bucket ^= 128u") == std::string::npos ||
      vulkan.find("kSortItemsPerThread = 2u") == std::string::npos ||
      vulkan.find("kSortRankGroupSize = 8u") == std::string::npos ||
      vulkan.find("kSortPackedWordCount = 8u") == std::string::npos ||
      vulkan.find("rund_sort_word_sum") == std::string::npos ||
      vulkan.find("rund_sort_group_prefix") == std::string::npos ||
      vulkan.find("atomicAdd(packed_counts") == std::string::npos ||
      dispatch.find("layout(local_size_x = 1) in") == std::string::npos ||
      dispatch.find("chunk < params.chunk_count") == std::string::npos ||
      dispatch.find("uint64_t(params.max_dispatch_groups)") ==
          std::string::npos ||
      dispatch.find("dispatch_args[chunk * 3u] = groups") ==
          std::string::npos) {
    return false;
  }
#endif
  return true;
}

} // namespace

bool BackendRunsScanSortCompact(const rund::AccelDevice &pick) {
  return pick.check.ok && CollectiveChunkMathIsExact() &&
         StatusDecodersAreCanonical() && ResidentShapeOverflowIsRejected() &&
         ScanSourceIsCanonical() && SignedSortSourcesCarryDomainOrder() &&
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
         SortMatchesCpuReference<rund::kernel::u32>(
             pick, rund::kernel::ComputeScalar::Lane32, SortU32Fixture()) &&
         SortMatchesCpuReference<rund::kernel::u32>(
             pick, rund::kernel::ComputeScalar::Lane32,
             std::array<rund::kernel::u32, 8u>{0u, 0xffffffffu, 2u, 0xfffffffdu,
                                               1u, 7u, 0xfffffff9u, 3u},
             0u, rund::kernel::ComputeDomain::I32) &&
         SortMatchesCpuReference<rund::kernel::u64>(
             pick, rund::kernel::ComputeScalar::Lane64, SortU64Fixture()) &&
         SortMatchesCpuReference<rund::kernel::u32>(
             pick, rund::kernel::ComputeScalar::Lane32,
             SortU32Declared16Fixture(), 16u) &&
         RadixBlocksRemainStable(pick) &&
         SortIdentityU32MatchesCpuReference(pick,
                                            rund::kernel::ComputeScalar::Lane32,
                                            SortU32Fixture(), 16u) &&
         CompactMatchesCpuReference(pick, rund::kernel::ComputeScalar::Lane32,
                                    CompactFixtureA(), 8u) &&
         CompactRejectsCapacityInsufficient(
             pick, rund::kernel::ComputeScalar::Lane32) &&
         bounded_sort::Contract(pick);
}

bool AvailableBackendsRunSameSort(const rund::AccelDevice &metal,
                                  const rund::AccelDevice &vulkan) {
  return metal.check.ok && metal.api == rund::AccelApi::Metal &&
         vulkan.check.ok && vulkan.api == rund::AccelApi::Vulkan &&
         SortBackendsAgree<rund::kernel::u32>(
             metal, vulkan, rund::kernel::ComputeScalar::Lane32,
             SortU32Fixture()) &&
         SortBackendsAgree<rund::kernel::u64>(
             metal, vulkan, rund::kernel::ComputeScalar::Lane64,
             SortU64Fixture()) &&
         CompactBackendsAgree(metal, vulkan,
                              rund::kernel::ComputeScalar::Lane32,
                              CompactFixtureA(), 8u);
}

} // namespace node_accel_contract::collective
