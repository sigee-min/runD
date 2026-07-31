#include "scratch.hpp"

#include "../context/internal.hpp"
#include "../primitive/block.hpp"
#include "../segmented/reduce/model.hpp"
#include "../sort/block/metal.hpp"
#include "../sort/block/vulkan.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

namespace rund::node::accel::detail {
namespace {

struct ScratchRequests final {
  std::array<std::uint64_t, 8u> bytes{};
  std::size_t count{};
  bool ok{true};

  void push(const std::uint64_t value) noexcept {
    if (value == 0u || count == bytes.size()) {
      ok = false;
      return;
    }
    bytes[count++] = value;
  }

  void product(const std::uint64_t left,
               const std::uint64_t right) noexcept {
    std::uint64_t value = 0u;
    if (!kernel::checked::mul(left, right, value)) {
      ok = false;
      return;
    }
    push(value);
  }
};

[[nodiscard]] ScratchRequests
scratch_requests(const Operation &operation, const rund::AccelApi api) noexcept {
  ScratchRequests result{};
  const bool metal = api == rund::AccelApi::Metal;
  const bool vulkan = api == rund::AccelApi::Vulkan;
  if (!metal && !vulkan) {
    return result;
  }
  switch (operation.kind()) {
  case rund::kernel::NodeKind::Scan: {
    const auto &plan = operation.get<operation::Scan>().plan;
    result.product(plan.block_count, plan.element_bytes);
    break;
  }
  case rund::kernel::NodeKind::SegmentedScan: {
    const auto &plan = operation.get<operation::SegmentedScan>().plan;
    result.product(plan.block_count, plan.element_bytes);
    result.product(plan.block_count, sizeof(rund::kernel::u32));
    break;
  }
  case rund::kernel::NodeKind::SegmentedReduce: {
    const auto &plan = operation.get<operation::SegmentedReduce>().plan;
    const SegmentedReduceLayout layout =
        SegmentedReduceLayoutFor(plan.element_count);
    const std::uint64_t index_bytes =
        metal ? sizeof(rund::kernel::u64) : sizeof(rund::kernel::u32);
    result.product(layout.block_count, index_bytes);
    result.product(layout.block_count, index_bytes);
    result.product(plan.element_count, index_bytes);
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
        !kernel::checked::mul(plan.bucket_count,
                              sizeof(rund::kernel::u32), buckets)) {
      result.ok = false;
      break;
    }
    result.push(plan.temp_key_bytes);
    result.push(plan.temp_value_bytes);
    if (metal) {
      result.push(table);
      result.push(table);
      result.push(buckets);
    } else {
      std::uint64_t counts = 0u;
      if (!kernel::checked::add(table, buckets, counts)) {
        result.ok = false;
        break;
      }
      result.push(counts);
      result.push(table);
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
      result.product(blocks, sizeof(rund::kernel::u32));
      result.product(blocks, sizeof(rund::kernel::u32));
      result.product(blocks, 32u * sizeof(rund::kernel::u32));
      result.product(kernel::checked::ceil(blocks, block_size),
                     sizeof(rund::kernel::u32));
    } else if (metal) {
      result.product(plan.element_count, sizeof(rund::kernel::u32));
      result.product(kernel::checked::ceil(plan.element_count, block_size),
                     sizeof(rund::kernel::u32));
    } else {
      result.product(blocks, sizeof(rund::kernel::u32));
      result.product(blocks, sizeof(rund::kernel::u32));
    }
    break;
  }
  case rund::kernel::NodeKind::Partition: {
    const auto &plan = operation.get<operation::Partition>().plan;
    const std::uint64_t block_size =
        metal ? block::MetalPartition : block::VulkanPartition;
    result.product(plan.element_count, sizeof(rund::kernel::u32));
    result.product(plan.element_count, sizeof(rund::kernel::u32));
    result.product(kernel::checked::ceil(plan.element_count, block_size),
                   sizeof(rund::kernel::u32));
    break;
  }
  case rund::kernel::NodeKind::Reduce: {
    const auto &plan = operation.get<operation::Reduce>().plan;
    result.push(plan.partial_bytes == 0u ? plan.partial_element_bytes
                                        : plan.partial_bytes);
    break;
  }
  case rund::kernel::NodeKind::ScatterReduce: {
    const auto &plan = operation.get<operation::ScatterReduce>().plan;
    result.product(plan.output_count, sizeof(rund::kernel::u32));
    break;
  }
  case rund::kernel::NodeKind::Map:
  case rund::kernel::NodeKind::Gather:
  case rund::kernel::NodeKind::Histogram:
  case rund::kernel::NodeKind::Scatter:
  case rund::kernel::NodeKind::Stencil:
  case rund::kernel::NodeKind::Transform:
  case rund::kernel::NodeKind::Matrix:
  case rund::kernel::NodeKind::Factor:
  case rund::kernel::NodeKind::Solve:
  case rund::kernel::NodeKind::Spectrum:
    break;
  }
  return result;
}

} // namespace

KernelScratchPlan PlanKernelScratch(const rund::AccelContext &context,
                                    const rund::AccelKernel &kernel,
                                    const std::uint64_t alignment,
                                    const std::uint64_t page_bytes) {
  if (alignment == 0u || (alignment & (alignment - 1u)) != 0u ||
      page_bytes == 0u || page_bytes % alignment != 0u) {
    return {};
  }
  const KernelExecution execution = AdmitKernelForExecution(context, kernel);
  if (!execution.admission.check.ok || execution.steps.empty()) {
    return KernelScratchPlan{.reason = execution.admission.check.reason};
  }
  struct Page final {
    std::uint64_t bytes{};
    std::uint64_t used{};
  };
  std::vector<Page> pages;
  std::size_t page_count = 0u;
  std::uint64_t last_bytes = 0u;
  for (const KernelExecutionStep &step : execution.steps) {
    pages.clear();
    const ScratchRequests requests =
        scratch_requests(step.operation, context.pick.api);
    if (!requests.ok) {
      return KernelScratchPlan{.reason = "compute_pipeline_capacity"};
    }
    for (std::size_t request = 0u; request < requests.count; ++request) {
      const std::uint64_t bytes = requests.bytes[request];
      if (bytes > page_bytes) {
        return KernelScratchPlan{.reason = "compute_resident_bytes_invalid"};
      }
      if (!scratch::fit(pages, alignment, bytes).ok) {
        pages.push_back(Page{.bytes = page_bytes, .used = bytes});
      }
    }
    if (pages.empty()) {
      continue;
    }
    std::uint64_t last = 0u;
    if (!scratch::align(pages.back().used, alignment, last)) {
      return KernelScratchPlan{.reason = "compute_pipeline_capacity"};
    }
    if (pages.size() > page_count) {
      page_count = pages.size();
      last_bytes = last;
    } else if (pages.size() == page_count) {
      last_bytes = std::max(last_bytes, last);
    }
  }
  if (page_count == 0u) {
    return KernelScratchPlan{.ok = true, .reason = "ok"};
  }
  return KernelScratchPlan{.last_bytes = last_bytes,
                           .page_count = page_count,
                           .ok = true,
                           .reason = "ok"};
}

bool ValidKernelScratch(const KernelScratchLayout &layout,
                        const RunBinds &binds) noexcept {
  if (layout.empty()) {
    return true;
  }
  if (!binds.valid() || binds.refs() == nullptr || binds.handles() == nullptr) {
    return false;
  }
  std::uint64_t prior_slot = 0u;
  for (std::size_t index = 0u; index < layout.size(); ++index) {
    const KernelScratchPage page = layout[index];
    if (page.bytes == 0u || page.slot >= binds.size() ||
        binds.handles()[page.slot] == nullptr ||
        binds.refs()[page.slot].offset_bytes > binds.refs()[page.slot].bytes ||
        page.bytes > binds.refs()[page.slot].bytes -
                         binds.refs()[page.slot].offset_bytes ||
        (index != 0u && page.slot <= prior_slot)) {
      return false;
    }
    prior_slot = page.slot;
  }
  return true;
}

} // namespace rund::node::accel::detail
