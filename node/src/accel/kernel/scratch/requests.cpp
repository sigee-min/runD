#include "internal.hpp"

#include "../../context/internal.hpp"
#include "../../primitive/block.hpp"
#include "../../scan/prefix.hpp"
#include "../../segmented/reduce/model.hpp"
#include "../../sort/block/metal.hpp"
#include "../../sort/block/vulkan.hpp"

#include <kernel/core/checked.hpp>

#include <optional>

namespace rund::node::accel::detail::kernel_scratch_internal {
namespace {

inline constexpr std::size_t kScratchRequestCapacity = 8u;

struct RequestBuilder final {
  ScratchRequests result{};

  void push(const std::uint64_t value) noexcept {
    if (value == 0u || result.count == kScratchRequestCapacity) {
      result.ok = false;
      return;
    }
    result.bytes[result.count++] = value;
  }

  void product(const std::uint64_t left, const std::uint64_t right) noexcept {
    std::uint64_t value = 0u;
    if (!kernel::checked::mul(left, right, value)) {
      result.ok = false;
      return;
    }
    push(value);
  }
};

} // namespace

ScratchRequests BuildScratchRequests(const Operation &operation,
                                     const rund::AccelApi api) noexcept {
  RequestBuilder builder{};
  const bool metal = api == rund::AccelApi::Metal;
  const bool vulkan = api == rund::AccelApi::Vulkan;
  if (!metal && !vulkan) {
    return builder.result;
  }
  switch (operation.kind()) {
  case rund::kernel::NodeKind::Scan: {
    const auto &plan = operation.get<operation::Scan>().plan;
    const std::optional<std::uint64_t> totals_bytes =
        ScanPrefixTotalsBytes(plan);
    if (!totals_bytes.has_value()) {
      builder.result.ok = false;
      break;
    }
    builder.push(*totals_bytes);
    break;
  }
  case rund::kernel::NodeKind::SegmentedScan: {
    const auto &plan = operation.get<operation::SegmentedScan>().plan;
    builder.product(plan.block_count, plan.element_bytes);
    builder.product(plan.block_count, sizeof(rund::kernel::u32));
    break;
  }
  case rund::kernel::NodeKind::SegmentedReduce: {
    const auto &plan = operation.get<operation::SegmentedReduce>().plan;
    const SegmentedReduceLayout layout =
        SegmentedReduceLayoutFor(plan.element_count);
    const std::uint64_t index_bytes =
        metal ? sizeof(rund::kernel::u64) : sizeof(rund::kernel::u32);
    builder.product(layout.block_count, index_bytes);
    builder.product(layout.block_count, index_bytes);
    builder.product(plan.element_count, index_bytes);
    break;
  }
  case rund::kernel::NodeKind::Sort: {
    const auto &plan = operation.get<operation::Sort>().plan;
    const std::uint64_t block_size =
        metal ? kMetalSortBlockSize : kVulkanSortBlockSize;
    const std::uint64_t blocks =
        kernel::checked::ceil(plan.element_count, block_size);
    std::uint64_t entries = 0u;
    std::uint64_t table = 0u;
    std::uint64_t buckets = 0u;
    if (!kernel::checked::mul(blocks, plan.bucket_count, entries) ||
        !kernel::checked::mul(entries, sizeof(rund::kernel::u32), table) ||
        !kernel::checked::mul(plan.bucket_count, sizeof(rund::kernel::u32),
                              buckets)) {
      builder.result.ok = false;
      break;
    }
    builder.push(plan.temp_key_bytes);
    builder.push(plan.temp_value_bytes);
    if (metal) {
      builder.push(table);
      builder.push(table);
      builder.push(buckets);
    } else {
      std::uint64_t counts = 0u;
      if (!kernel::checked::add(table, buckets, counts)) {
        builder.result.ok = false;
        break;
      }
      builder.push(counts);
      builder.push(table);
    }
    break;
  }
  case rund::kernel::NodeKind::Compact: {
    const auto &plan = operation.get<operation::Compact>().plan;
    const std::uint64_t block_size =
        metal ? block::MetalCompact : block::VulkanCompact;
    const std::uint64_t blocks =
        kernel::checked::ceil(plan.element_count, block_size);
    if (metal && plan.status_bytes == 0u) {
      builder.product(blocks, sizeof(rund::kernel::u32));
      builder.product(blocks, sizeof(rund::kernel::u32));
      builder.product(blocks, 32u * sizeof(rund::kernel::u32));
      builder.product(kernel::checked::ceil(blocks, block_size),
                      sizeof(rund::kernel::u32));
    } else if (metal) {
      builder.product(plan.element_count, sizeof(rund::kernel::u32));
      builder.product(kernel::checked::ceil(plan.element_count, block_size),
                      sizeof(rund::kernel::u32));
    } else {
      builder.product(blocks, sizeof(rund::kernel::u32));
      builder.product(blocks, sizeof(rund::kernel::u32));
    }
    break;
  }
  case rund::kernel::NodeKind::Partition: {
    const auto &plan = operation.get<operation::Partition>().plan;
    const std::uint64_t block_size =
        metal ? block::MetalPartition : block::VulkanPartition;
    builder.product(plan.element_count, sizeof(rund::kernel::u32));
    builder.product(plan.element_count, sizeof(rund::kernel::u32));
    builder.product(kernel::checked::ceil(plan.element_count, block_size),
                    sizeof(rund::kernel::u32));
    break;
  }
  case rund::kernel::NodeKind::Reduce: {
    const auto &plan = operation.get<operation::Reduce>().plan;
    builder.push(plan.partial_bytes == 0u ? plan.partial_element_bytes
                                          : plan.partial_bytes);
    break;
  }
  case rund::kernel::NodeKind::ScatterReduce: {
    const auto &plan = operation.get<operation::ScatterReduce>().plan;
    builder.product(plan.output_count, sizeof(rund::kernel::u32));
    break;
  }
  case rund::kernel::NodeKind::Map:
  case rund::kernel::NodeKind::Gather:
  case rund::kernel::NodeKind::Histogram:
  case rund::kernel::NodeKind::Scatter:
  case rund::kernel::NodeKind::Stencil:
  case rund::kernel::NodeKind::Window:
  case rund::kernel::NodeKind::Transform:
  case rund::kernel::NodeKind::Matrix:
  case rund::kernel::NodeKind::Factor:
  case rund::kernel::NodeKind::Solve:
  case rund::kernel::NodeKind::Spectrum:
    break;
  }
  return builder.result;
}

} // namespace rund::node::accel::detail::kernel_scratch_internal
