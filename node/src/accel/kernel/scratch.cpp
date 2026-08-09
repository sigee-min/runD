#include "scratch.hpp"

#include "../context/internal.hpp"
#include "../primitive/block.hpp"
#include "../range_aggregate/model.hpp"
#include "../scan/prefix.hpp"
#include "../segmented/reduce/model.hpp"
#include "../sort/block/metal.hpp"
#include "../sort/block/vulkan.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <new>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rund::node::accel::detail {
namespace {

inline constexpr std::uint32_t kRangeScratchRoleTag = 0x52410000u;

[[nodiscard]] constexpr std::uint32_t
range_role_value(const RangeTempRole role) noexcept {
  switch (role) {
  case RangeTempRole::PrefixValues:
    return 0u;
  case RangeTempRole::BlockSummaries:
    return 1u;
  case RangeTempRole::ForwardValues:
    return 2u;
  case RangeTempRole::BackwardValues:
    return 3u;
  }
  return std::numeric_limits<std::uint32_t>::max();
}

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

  void product(const std::uint64_t left, const std::uint64_t right) noexcept {
    std::uint64_t value = 0u;
    if (!kernel::checked::mul(left, right, value)) {
      ok = false;
      return;
    }
    push(value);
  }
};

[[nodiscard]] ScratchRequests
scratch_requests(const Operation &operation,
                 const rund::AccelApi api) noexcept {
  ScratchRequests result{};
  const bool metal = api == rund::AccelApi::Metal;
  const bool vulkan = api == rund::AccelApi::Vulkan;
  if (!metal && !vulkan) {
    return result;
  }
  switch (operation.kind()) {
  case rund::kernel::NodeKind::Scan: {
    const auto &plan = operation.get<operation::Scan>().plan;
    const std::optional<std::uint64_t> totals_bytes =
        ScanPrefixTotalsBytes(plan);
    if (!totals_bytes.has_value()) {
      result.ok = false;
      break;
    }
    result.push(*totals_bytes);
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
        !kernel::checked::mul(plan.bucket_count, sizeof(rund::kernel::u32),
                              buckets)) {
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
  case rund::kernel::NodeKind::Window:
  case rund::kernel::NodeKind::Transform:
  case rund::kernel::NodeKind::Matrix:
  case rund::kernel::NodeKind::Factor:
  case rund::kernel::NodeKind::Solve:
  case rund::kernel::NodeKind::Spectrum:
    break;
  }
  return result;
}

[[nodiscard]] constexpr bool
lifetimes_overlap(const KernelScratchRequirement &left,
                  const KernelScratchRequirement &right) noexcept {
  return left.first_stage <= right.last_stage &&
         right.first_stage <= left.last_stage;
}

[[nodiscard]] constexpr bool
ranges_overlap(const std::uint64_t left_offset, const std::uint64_t left_end,
               const std::uint64_t right_offset,
               const std::uint64_t right_end) noexcept {
  return left_offset < right_end && right_offset < left_end;
}

[[nodiscard]] bool
canonical_requirement_less(const KernelScratchRequirement &left,
                           const KernelScratchRequirement &right) noexcept {
  if (left.first_stage != right.first_stage) {
    return left.first_stage < right.first_stage;
  }
  if (left.role != right.role) {
    return left.role.value() < right.role.value();
  }
  return left.last_stage < right.last_stage;
}

[[nodiscard]] bool validate_requirements(
    const std::span<const KernelScratchRequirement> requirements,
    const std::uint64_t backing_alignment) noexcept {
  for (std::size_t index = 0u; index < requirements.size(); ++index) {
    const KernelScratchRequirement &requirement = requirements[index];
    if (!requirement.valid() ||
        backing_alignment % requirement.alignment != 0u) {
      return false;
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (requirements[prior].role == requirement.role) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool
logical_payload(const std::span<const KernelScratchRequirement> requirements,
                std::uint64_t &result) noexcept {
  result = 0u;
  for (const KernelScratchRequirement &frontier : requirements) {
    std::uint64_t live = 0u;
    for (const KernelScratchRequirement &requirement : requirements) {
      if (requirement.first_stage <= frontier.first_stage &&
          frontier.first_stage <= requirement.last_stage &&
          !kernel::checked::add(live, requirement.bytes, live)) {
        return false;
      }
    }
    result = std::max(result, live);
  }
  return true;
}

[[nodiscard]] bool placement_offset(
    const std::vector<KernelScratchPlacement> &placements,
    const std::size_t page, const KernelScratchRequirement &requirement,
    const std::uint64_t backing_alignment, const std::uint64_t page_bytes,
    std::uint64_t &result) noexcept {
  std::uint64_t cursor = 0u;
  for (;;) {
    std::uint64_t offset = 0u;
    if (!scratch::align(cursor, backing_alignment, offset) ||
        offset > page_bytes || requirement.bytes > page_bytes - offset) {
      return false;
    }
    const std::uint64_t end = offset + requirement.bytes;
    const KernelScratchPlacement *blocker = nullptr;
    std::uint64_t blocker_end = 0u;
    for (const KernelScratchPlacement &placed : placements) {
      if (placed.page != page ||
          !lifetimes_overlap(requirement, placed.requirement)) {
        continue;
      }
      const std::uint64_t placed_end = placed.offset + placed.requirement.bytes;
      if (!ranges_overlap(offset, end, placed.offset, placed_end)) {
        continue;
      }
      if (blocker == nullptr || placed.offset < blocker->offset ||
          (placed.offset == blocker->offset && placed_end > blocker_end)) {
        blocker = &placed;
        blocker_end = placed_end;
      }
    }
    if (blocker == nullptr) {
      result = offset;
      return true;
    }
    cursor = blocker_end;
  }
}

} // namespace

KernelScratchBatchPlan::KernelScratchBatchPlan(
    const State state, const char *const reason) noexcept
    : state_(state), reason_(reason) {}

KernelScratchBatchPlan
KernelScratchBatchPlan::failure(const char *const reason) noexcept {
  return KernelScratchBatchPlan{State::Failed, reason};
}

KernelScratchBatchPlan KernelScratchBatchPlan::success(
    std::vector<KernelScratchPlacement> placements,
    const std::uint64_t payload_bytes, const std::uint64_t backing_bytes,
    const std::uint64_t last_bytes, const std::size_t page_count) noexcept {
  KernelScratchBatchPlan result{State::Ready, "ok"};
  result.placements_ = std::move(placements);
  result.payload_bytes_ = payload_bytes;
  result.backing_bytes_ = backing_bytes;
  result.last_bytes_ = last_bytes;
  result.page_count_ = page_count;
  return result;
}

bool KernelScratchBatchPlan::ok() const noexcept {
  return state_ == State::Ready;
}

const char *KernelScratchBatchPlan::reason() const noexcept { return reason_; }

const std::vector<KernelScratchPlacement> &
KernelScratchBatchPlan::placements() const noexcept {
  return placements_;
}

std::uint64_t KernelScratchBatchPlan::payload_bytes() const noexcept {
  return payload_bytes_;
}

std::uint64_t KernelScratchBatchPlan::backing_bytes() const noexcept {
  return backing_bytes_;
}

std::uint64_t KernelScratchBatchPlan::last_bytes() const noexcept {
  return last_bytes_;
}

std::size_t KernelScratchBatchPlan::page_count() const noexcept {
  return page_count_;
}

KernelScratchBatchPlan PlanKernelScratchBatch(
    const std::span<const KernelScratchRequirement> requirements,
    const std::uint64_t backing_alignment, const std::uint64_t page_bytes) {
  if (backing_alignment == 0u ||
      (backing_alignment & (backing_alignment - 1u)) != 0u ||
      page_bytes == 0u || page_bytes % backing_alignment != 0u ||
      !validate_requirements(requirements, backing_alignment)) {
    return KernelScratchBatchPlan::failure("accel_kernel_scratch_invalid");
  }
  if (requirements.empty()) {
    return KernelScratchBatchPlan::success({}, 0u, 0u, 0u, 0u);
  }
  for (const KernelScratchRequirement &requirement : requirements) {
    if (requirement.bytes > page_bytes) {
      return KernelScratchBatchPlan::failure("compute_resident_bytes_invalid");
    }
  }

  std::uint64_t payload_bytes = 0u;
  if (!logical_payload(requirements, payload_bytes)) {
    return KernelScratchBatchPlan::failure("compute_pipeline_capacity");
  }

  struct Page final {
    std::uint64_t extent{};
  };
  try {
    std::vector<std::size_t> order(requirements.size());
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(),
              [&](const std::size_t left, const std::size_t right) {
                return canonical_requirement_less(requirements[left],
                                                  requirements[right]);
              });

    std::vector<Page> pages;
    pages.reserve(requirements.size());
    std::vector<KernelScratchPlacement> placements;
    placements.reserve(requirements.size());
    for (const std::size_t ordinal : order) {
      const KernelScratchRequirement &requirement = requirements[ordinal];
      bool placed = false;
      for (std::size_t page = 0u; page < pages.size(); ++page) {
        std::uint64_t offset = 0u;
        if (!placement_offset(placements, page, requirement, backing_alignment,
                              page_bytes, offset)) {
          continue;
        }
        pages[page].extent =
            std::max(pages[page].extent, offset + requirement.bytes);
        placements.push_back(KernelScratchPlacement{
            .requirement = requirement,
            .page = page,
            .offset = offset,
        });
        placed = true;
        break;
      }
      if (!placed) {
        pages.push_back(Page{.extent = requirement.bytes});
        placements.push_back(KernelScratchPlacement{
            .requirement = requirement,
            .page = pages.size() - 1u,
        });
      }
    }

    std::uint64_t last_bytes = 0u;
    if (pages.empty() ||
        !scratch::align(pages.back().extent, backing_alignment, last_bytes) ||
        last_bytes == 0u || last_bytes > page_bytes ||
        pages.size() - 1u > static_cast<std::size_t>(
                                std::numeric_limits<std::uint64_t>::max())) {
      return KernelScratchBatchPlan::failure("compute_pipeline_capacity");
    }
    std::uint64_t leading_bytes = 0u;
    std::uint64_t backing_bytes = 0u;
    if (!kernel::checked::mul(static_cast<std::uint64_t>(pages.size() - 1u),
                              page_bytes, leading_bytes) ||
        !kernel::checked::add(leading_bytes, last_bytes, backing_bytes) ||
        payload_bytes > backing_bytes) {
      return KernelScratchBatchPlan::failure("compute_pipeline_capacity");
    }
    return KernelScratchBatchPlan::success(std::move(placements), payload_bytes,
                                           backing_bytes, last_bytes,
                                           pages.size());
  } catch (const std::bad_alloc &) {
    return KernelScratchBatchPlan::failure("compute_pipeline_capacity");
  } catch (const std::length_error &) {
    return KernelScratchBatchPlan::failure("compute_pipeline_capacity");
  }
}

const KernelScratchPlacement *
FindKernelScratchPlacement(const KernelScratchBatchPlan &plan,
                           const KernelScratchRole role) noexcept {
  if (!plan.ok() || !role.valid()) {
    return nullptr;
  }
  const auto found =
      std::find_if(plan.placements().begin(), plan.placements().end(),
                   [&](const KernelScratchPlacement &placement) {
                     return placement.requirement.role == role;
                   });
  return found == plan.placements().end() ? nullptr : &*found;
}

KernelScratchRole ScratchRoleForRangeTemp(const RangeTempRole role,
                                          const std::uint8_t ordinal) noexcept {
  const std::uint32_t value = range_role_value(role);
  return value == std::numeric_limits<std::uint32_t>::max()
             ? KernelScratchRole{}
             : KernelScratchRole::from(kRangeScratchRoleTag | (value << 8u) |
                                       ordinal);
}

KernelScratchRequirement
ScratchReqForRangeTemp(const RangeTempReq &requirement) noexcept {
  return KernelScratchRequirement{
      .role = ScratchRoleForRangeTemp(requirement.role, requirement.ordinal),
      .bytes = requirement.bytes,
      .alignment = requirement.alignment,
      .first_stage = requirement.first_stage,
      .last_stage = requirement.last_stage,
  };
}

KernelScratchBatchPlan PlanRangeScratch(const RangePlan &plan,
                                        const std::uint64_t backing_alignment,
                                        const std::uint64_t page_bytes) {
  if (!plan.ok()) {
    return KernelScratchBatchPlan::failure(plan.reason());
  }
  const std::size_t stage_count = plan.stage_count();
  const std::size_t temporary_count = plan.temporary_count();
  if (stage_count == 0u || stage_count > kRangeStageCap ||
      temporary_count > kRangeTempCap) {
    return KernelScratchBatchPlan::failure("accel_kernel_scratch_invalid");
  }
  std::array<KernelScratchRequirement, kRangeTempCap> requirements{};
  for (std::size_t index = 0u; index < temporary_count; ++index) {
    const RangeTempReq temporary = plan.temporary(index);
    requirements[index] = ScratchReqForRangeTemp(temporary);
    if (!requirements[index].valid() ||
        requirements[index].last_stage >= stage_count) {
      return KernelScratchBatchPlan::failure("accel_kernel_scratch_invalid");
    }
  }
  return PlanKernelScratchBatch(
      std::span<const KernelScratchRequirement>{requirements.data(),
                                                temporary_count},
      backing_alignment, page_bytes);
}

const KernelScratchPlacement *
FindRangeScratch(const KernelScratchBatchPlan &plan, const RangeTempRole role,
                 const std::uint8_t ordinal) noexcept {
  return FindKernelScratchPlacement(plan,
                                    ScratchRoleForRangeTemp(role, ordinal));
}

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
  std::size_t page_count = 0u;
  std::uint64_t last_bytes = 0u;
  std::uint64_t backing_bytes = 0u;
  std::uint64_t payload_bytes = 0u;
  for (const KernelExecutionStep &step : execution.steps) {
    if (const RangePlan *const range = RangePlanFor(step.operation);
        range != nullptr) {
      const KernelScratchBatchPlan batch =
          PlanRangeScratch(*range, alignment, page_bytes);
      if (!batch.ok()) {
        return KernelScratchPlan{.reason = batch.reason()};
      }
      payload_bytes = std::max(payload_bytes, batch.payload_bytes());
      if (batch.page_count() > page_count ||
          (batch.page_count() == page_count &&
           batch.last_bytes() > last_bytes)) {
        page_count = batch.page_count();
        last_bytes = batch.last_bytes();
        backing_bytes = batch.backing_bytes();
      }
      continue;
    }
    const ScratchRequests requests =
        scratch_requests(step.operation, context.pick.api);
    if (!requests.ok) {
      return KernelScratchPlan{.reason = "compute_pipeline_capacity"};
    }
    std::array<KernelScratchRequirement, 8u> requirements{};
    for (std::size_t request = 0u; request < requests.count; ++request) {
      const std::uint64_t bytes = requests.bytes[request];
      requirements[request] = KernelScratchRequirement{
          .role = KernelScratchRole::from(static_cast<std::uint32_t>(request)),
          .bytes = bytes,
          .alignment = alignment,
      };
    }
    const KernelScratchBatchPlan batch = PlanKernelScratchBatch(
        std::span<const KernelScratchRequirement>{requirements.data(),
                                                  requests.count},
        alignment, page_bytes);
    if (!batch.ok()) {
      return KernelScratchPlan{.reason = batch.reason()};
    }
    payload_bytes = std::max(payload_bytes, batch.payload_bytes());
    if (batch.page_count() == 0u) {
      continue;
    }
    if (batch.page_count() > page_count ||
        (batch.page_count() == page_count && batch.last_bytes() > last_bytes)) {
      page_count = batch.page_count();
      last_bytes = batch.last_bytes();
      backing_bytes = batch.backing_bytes();
    }
  }
  if (page_count == 0u) {
    return KernelScratchPlan{
        .payload_bytes = payload_bytes,
        .ok = true,
        .reason = "ok",
    };
  }
  return KernelScratchPlan{.payload_bytes = payload_bytes,
                           .backing_bytes = backing_bytes,
                           .last_bytes = last_bytes,
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
