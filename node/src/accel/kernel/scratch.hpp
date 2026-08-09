#pragma once

#include "bindings/refs.hpp"

#include <accel/context/value.hpp>
#include <accel/kernel/value.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace rund::node::accel::detail {

class RangePlan;
struct RangeTempReq;
enum class RangeTempRole : std::uint8_t;

class KernelScratchRole final {
public:
  constexpr KernelScratchRole() noexcept = default;

  [[nodiscard]] static constexpr KernelScratchRole
  from(const std::uint32_t value) noexcept {
    return value == Invalid ? KernelScratchRole{} : KernelScratchRole{value};
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return value_ != Invalid;
  }

  [[nodiscard]] constexpr std::uint32_t value() const noexcept {
    return value_;
  }

  friend constexpr bool
  operator==(const KernelScratchRole &,
             const KernelScratchRole &) noexcept = default;

private:
  static constexpr std::uint32_t Invalid =
      std::numeric_limits<std::uint32_t>::max();

  explicit constexpr KernelScratchRole(const std::uint32_t value) noexcept
      : value_(value) {}

  std::uint32_t value_{Invalid};
};

// A source-private algorithm planner describes logical temporary roles only.
// This descriptor is the adapter boundary into the one physical Pipeline
// scratch authority. Stage lifetimes are closed and alignment is a required
// byte alignment, not retained padding.
struct KernelScratchRequirement final {
  KernelScratchRole role{};
  std::uint64_t bytes{};
  std::uint64_t alignment{};
  std::uint32_t first_stage{};
  std::uint32_t last_stage{};

  [[nodiscard]] constexpr bool valid() const noexcept {
    return role.valid() && bytes != 0u && alignment != 0u &&
           (alignment & (alignment - 1u)) == 0u && first_stage <= last_stage;
  }

  friend constexpr bool
  operator==(const KernelScratchRequirement &,
             const KernelScratchRequirement &) noexcept = default;
};

struct KernelScratchPlacement final {
  KernelScratchRequirement requirement{};
  std::size_t page{};
  std::uint64_t offset{};

  [[nodiscard]] constexpr bool valid() const noexcept {
    return requirement.valid();
  }

  friend constexpr bool
  operator==(const KernelScratchPlacement &,
             const KernelScratchPlacement &) noexcept = default;
};

static_assert(std::is_trivially_copyable_v<KernelScratchRole>);
static_assert(std::is_trivially_copyable_v<KernelScratchRequirement>);
static_assert(std::is_trivially_copyable_v<KernelScratchPlacement>);

class KernelScratchBatchPlan final {
public:
  [[nodiscard]] bool ok() const noexcept;
  [[nodiscard]] const char *reason() const noexcept;
  [[nodiscard]] const std::vector<KernelScratchPlacement> &
  placements() const noexcept;
  // Exact maximum simultaneously-live logical role payload. This excludes
  // alignment holes and inactive serial stages.
  [[nodiscard]] std::uint64_t payload_bytes() const noexcept;
  // Exact retained backing under the Pipeline page convention: every page
  // before the last is full and the last ends at last_bytes.
  [[nodiscard]] std::uint64_t backing_bytes() const noexcept;
  [[nodiscard]] std::uint64_t last_bytes() const noexcept;
  [[nodiscard]] std::size_t page_count() const noexcept;

private:
  friend KernelScratchBatchPlan
  PlanKernelScratchBatch(std::span<const KernelScratchRequirement> requirements,
                         std::uint64_t backing_alignment,
                         std::uint64_t page_bytes);
  friend KernelScratchBatchPlan
  PlanRangeScratch(const RangePlan &plan, std::uint64_t backing_alignment,
                   std::uint64_t page_bytes);

  enum class State : std::uint8_t {
    Failed,
    Ready,
  };

  explicit KernelScratchBatchPlan(State state, const char *reason) noexcept;

  [[nodiscard]] static KernelScratchBatchPlan
  failure(const char *reason) noexcept;

  [[nodiscard]] static KernelScratchBatchPlan
  success(std::vector<KernelScratchPlacement> placements,
          std::uint64_t payload_bytes, std::uint64_t backing_bytes,
          std::uint64_t last_bytes, std::size_t page_count) noexcept;

  std::vector<KernelScratchPlacement> placements_{};
  std::uint64_t payload_bytes_{};
  std::uint64_t backing_bytes_{};
  std::uint64_t last_bytes_{};
  std::size_t page_count_{};
  State state_{State::Failed};
  const char *reason_{"accel_kernel_scratch_invalid"};
};

struct KernelScratchPlan final {
  std::uint64_t payload_bytes{};
  std::uint64_t backing_bytes{};
  std::uint64_t last_bytes{};
  std::size_t page_count{};
  bool ok{};
  const char *reason{"accel_kernel_scratch_invalid"};
};

struct KernelScratchPage final {
  std::size_t slot{};
  std::uint64_t bytes{};
};

using KernelScratchLayout = std::vector<KernelScratchPage>;

namespace scratch {

struct Placement final {
  std::size_t page{};
  std::uint64_t offset{};
  bool ok{};
};

[[nodiscard]] inline bool align(const std::uint64_t value,
                                const std::uint64_t alignment,
                                std::uint64_t &result) noexcept {
  if (alignment == 0u || (alignment & (alignment - 1u)) != 0u) {
    return false;
  }
  const std::uint64_t mask = alignment - 1u;
  if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
    return false;
  }
  result = (value + mask) & ~mask;
  return true;
}

template <class Pages>
[[nodiscard]] Placement fit(Pages &pages, const std::uint64_t alignment,
                            const std::uint64_t bytes) noexcept {
  if (bytes == 0u) {
    return {};
  }
  for (std::size_t index = 0u; index < pages.size(); ++index) {
    auto &page = pages[index];
    std::uint64_t offset = 0u;
    if (!align(page.used, alignment, offset) || offset > page.bytes ||
        bytes > page.bytes - offset) {
      continue;
    }
    page.used = offset + bytes;
    return Placement{.page = index, .offset = offset, .ok = true};
  }
  return {};
}

template <class Pages> void reset(Pages &pages) noexcept {
  for (auto &page : pages) {
    page.used = 0u;
  }
}

template <class Pages> [[nodiscard]] bool active(const Pages &pages) noexcept {
  for (const auto &page : pages) {
    if (page.used != 0u) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] inline bool valid(const KernelScratchPlacement &placement,
                                const std::uint64_t backing_alignment,
                                const std::uint64_t page_bytes) noexcept {
  return placement.valid() && backing_alignment != 0u &&
         (backing_alignment & (backing_alignment - 1u)) == 0u &&
         backing_alignment % placement.requirement.alignment == 0u &&
         placement.offset % backing_alignment == 0u &&
         placement.offset <= page_bytes &&
         placement.requirement.bytes <= page_bytes - placement.offset;
}

} // namespace scratch

[[nodiscard]] KernelScratchBatchPlan
PlanKernelScratchBatch(std::span<const KernelScratchRequirement> requirements,
                       std::uint64_t backing_alignment,
                       std::uint64_t page_bytes);

[[nodiscard]] const KernelScratchPlacement *
FindKernelScratchPlacement(const KernelScratchBatchPlan &plan,
                           KernelScratchRole role) noexcept;

[[nodiscard]] KernelScratchRole
ScratchRoleForRangeTemp(RangeTempRole role, std::uint8_t ordinal) noexcept;

[[nodiscard]] KernelScratchRequirement
ScratchReqForRangeTemp(const RangeTempReq &requirement) noexcept;

[[nodiscard]] KernelScratchBatchPlan
PlanRangeScratch(const RangePlan &plan, std::uint64_t backing_alignment,
                 std::uint64_t page_bytes);

[[nodiscard]] const KernelScratchPlacement *
FindRangeScratch(const KernelScratchBatchPlan &plan, RangeTempRole role,
                 std::uint8_t ordinal) noexcept;

[[nodiscard]] KernelScratchPlan
PlanKernelScratch(const rund::AccelContext &context,
                  const rund::AccelKernel &kernel, std::uint64_t alignment,
                  std::uint64_t page_bytes);

[[nodiscard]] bool ValidKernelScratch(const KernelScratchLayout &layout,
                                      const RunBinds &binds) noexcept;

} // namespace rund::node::accel::detail
