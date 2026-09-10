#include "../scratch.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <limits>
#include <new>
#include <numeric>
#include <utility>

namespace rund::node::accel::detail {
namespace {

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

} // namespace rund::node::accel::detail
