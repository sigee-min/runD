#pragma once

#include "../kernel/bindings/stencil.hpp"
#include "../primitive/shape.hpp"

#include <kernel/program/compute/stencil/plan.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace rund::node::accel::detail {

inline constexpr std::array<rund::kernel::u32, 3u> kStencilPhysicalGroupWidths{
    64u, 128u, 256u};
inline constexpr rund::kernel::u32 kStencilSharedMemoryOccupancyBudget = 4u;

class StencilGpuShape final {
public:
  constexpr StencilGpuShape() noexcept = default;

  [[nodiscard]] static constexpr StencilGpuShape
  direct(const rund::kernel::u32 width) noexcept {
    return SupportedWidth(width) ? StencilGpuShape{width, 0u}
                                 : StencilGpuShape{};
  }

  [[nodiscard]] static constexpr StencilGpuShape
  shared(const rund::kernel::u32 width,
         const rund::kernel::u32 radius_cap) noexcept {
    return SupportedWidth(width) && radius_cap != 0u && radius_cap <= width
               ? StencilGpuShape{width, radius_cap}
               : StencilGpuShape{};
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return SupportedWidth(width_) && radius_cap_ <= width_;
  }

  [[nodiscard]] constexpr bool uses_shared_memory() const noexcept {
    return valid() && radius_cap_ != 0u;
  }

  [[nodiscard]] constexpr rund::kernel::u32 width() const noexcept {
    return width_;
  }

  [[nodiscard]] constexpr rund::kernel::u32 radius_cap() const noexcept {
    return radius_cap_;
  }

  [[nodiscard]] constexpr rund::kernel::u32
  shared_element_capacity() const noexcept {
    return uses_shared_memory() ? width_ + 2u * radius_cap_ : 0u;
  }

  [[nodiscard]] constexpr rund::kernel::u64
  shared_bytes(const rund::kernel::u32 element_bytes) const noexcept {
    return static_cast<rund::kernel::u64>(shared_element_capacity()) *
           element_bytes;
  }

  [[nodiscard]] constexpr std::uint64_t identity() const noexcept {
    return (static_cast<std::uint64_t>(width_) << 32u) | radius_cap_;
  }

  friend constexpr bool operator==(const StencilGpuShape &,
                                   const StencilGpuShape &) noexcept = default;

private:
  constexpr StencilGpuShape(const rund::kernel::u32 width,
                            const rund::kernel::u32 radius_cap) noexcept
      : width_(width), radius_cap_(radius_cap) {}

  [[nodiscard]] static constexpr bool
  SupportedWidth(const rund::kernel::u32 width) noexcept {
    return width == 64u || width == 128u || width == 256u;
  }

  rund::kernel::u32 width_{};
  rund::kernel::u32 radius_cap_{};
};

static_assert(sizeof(StencilGpuShape) == 8u);
static_assert(std::is_trivially_copyable_v<StencilGpuShape>);

struct StencilGpuCapabilities final {
  rund::kernel::u32 maximum_workgroup_width{};
  rund::kernel::u32 shared_memory_occupancy_budget{};
  rund::kernel::u64 shared_memory_limit{};
  rund::kernel::u64 maximum_group_count{};
};

[[nodiscard]] constexpr rund::kernel::u32
StencilElementBytes(const rund::kernel::StencilElement element) noexcept {
  if (element == rund::kernel::StencilElement::U32) {
    return 4u;
  }
  return element == rund::kernel::StencilElement::U64 ? 8u : 0u;
}

[[nodiscard]] constexpr rund::kernel::u32 StencilSharedRadiusCapacity(
    const rund::kernel::u32 width, const rund::kernel::u32 element_bytes,
    const StencilGpuCapabilities capabilities) noexcept {
  if ((element_bytes != 4u && element_bytes != 8u) ||
      capabilities.shared_memory_occupancy_budget == 0u ||
      width > capabilities.maximum_workgroup_width) {
    return 0u;
  }
  const rund::kernel::u64 per_group =
      capabilities.shared_memory_limit /
      capabilities.shared_memory_occupancy_budget;
  const rund::kernel::u64 slots = per_group / element_bytes;
  if (slots <= width) {
    return 0u;
  }
  return static_cast<rund::kernel::u32>(
      std::min<rund::kernel::u64>(width, (slots - width) / 2u));
}

[[nodiscard]] constexpr rund::kernel::u64
StencilPhysicalGroupCount(const rund::kernel::u64 element_count,
                          const StencilGpuShape shape) noexcept {
  return !shape.valid() || element_count == 0u
             ? 0u
             : 1u + (element_count - 1u) / shape.width();
}

namespace stencil_shape_detail {

struct Score final {
  bool direct{};
  rund::kernel::u64 global_reads{};
  rund::kernel::u64 launched_lanes{};
  rund::kernel::u64 groups{};
  rund::kernel::u64 shared_bytes{};
  rund::kernel::u32 width{};
};

struct RankedCandidate final {
  StencilGpuShape shape{};
  Score score{};
};

[[nodiscard]] constexpr bool Less(const Score left,
                                  const Score right) noexcept {
  if (left.direct != right.direct) {
    return !left.direct;
  }
  if (left.global_reads != right.global_reads) {
    return left.global_reads < right.global_reads;
  }
  if (left.launched_lanes != right.launched_lanes) {
    return left.launched_lanes < right.launched_lanes;
  }
  if (left.groups != right.groups) {
    return left.groups < right.groups;
  }
  if (left.shared_bytes != right.shared_bytes) {
    return left.shared_bytes < right.shared_bytes;
  }
  return left.width < right.width;
}

[[nodiscard]] constexpr rund::kernel::u64
SharedGlobalReads(const rund::kernel::u64 element_count,
                  const rund::kernel::u64 groups, const rund::kernel::u32 width,
                  const rund::kernel::u32 radius) noexcept {
  if (groups <= 1u) {
    return element_count;
  }
  const rund::kernel::u64 tail =
      element_count - (groups - 1u) * static_cast<rund::kernel::u64>(width);
  return element_count + (2u * groups - 3u) * radius +
         std::min<rund::kernel::u64>(radius, tail);
}

inline constexpr std::size_t kCandidateCapacity =
    2u * kStencilPhysicalGroupWidths.size();

constexpr void
Insert(std::array<RankedCandidate, kCandidateCapacity> &candidates,
       std::size_t &count, const RankedCandidate candidate) noexcept {
  std::size_t position = count;
  while (position != 0u &&
         Less(candidate.score, candidates[position - 1u].score)) {
    candidates[position] = candidates[position - 1u];
    --position;
  }
  candidates[position] = candidate;
  ++count;
}

} // namespace stencil_shape_detail

struct StencilGpuShapeCandidates final {
  std::array<StencilGpuShape, stencil_shape_detail::kCandidateCapacity>
      values{};
  std::size_t count{};

  [[nodiscard]] constexpr bool empty() const noexcept { return count == 0u; }
  [[nodiscard]] constexpr std::size_t size() const noexcept { return count; }

  [[nodiscard]] constexpr const StencilGpuShape &
  operator[](const std::size_t index) const noexcept {
    return values[index];
  }

  [[nodiscard]] constexpr const StencilGpuShape *begin() const noexcept {
    return values.data();
  }

  [[nodiscard]] constexpr const StencilGpuShape *end() const noexcept {
    return values.data() + count;
  }
};

[[nodiscard]] constexpr StencilGpuShapeCandidates
RankStencilGpuShapes(const rund::kernel::u64 element_count,
                     const rund::kernel::u64 radius,
                     const rund::kernel::u32 element_bytes,
                     const StencilGpuCapabilities capabilities) noexcept {
  StencilGpuShapeCandidates result{};
  if (element_count == 0u || radius == 0u || radius > element_count ||
      (element_bytes != 4u && element_bytes != 8u) ||
      capabilities.maximum_group_count == 0u ||
      capabilities.shared_memory_occupancy_budget == 0u) {
    return result;
  }

  std::array<stencil_shape_detail::RankedCandidate,
             stencil_shape_detail::kCandidateCapacity>
      ranked{};
  std::size_t count = 0u;
  for (const rund::kernel::u32 width : kStencilPhysicalGroupWidths) {
    if (width > capabilities.maximum_workgroup_width) {
      continue;
    }
    const StencilGpuShape direct = StencilGpuShape::direct(width);
    const rund::kernel::u64 groups =
        StencilPhysicalGroupCount(element_count, direct);
    if (groups == 0u || groups > capabilities.maximum_group_count ||
        groups > std::numeric_limits<rund::kernel::u32>::max()) {
      continue;
    }
    const rund::kernel::u32 radius_cap =
        StencilSharedRadiusCapacity(width, element_bytes, capabilities);
    if (radius <= radius_cap) {
      const StencilGpuShape shared = StencilGpuShape::shared(width, radius_cap);
      stencil_shape_detail::Insert(
          ranked, count,
          stencil_shape_detail::RankedCandidate{
              .shape = shared,
              .score =
                  {
                      .direct = false,
                      .global_reads = stencil_shape_detail::SharedGlobalReads(
                          element_count, groups, width,
                          static_cast<rund::kernel::u32>(radius)),
                      .launched_lanes = groups * width,
                      .groups = groups,
                      .shared_bytes = shared.shared_bytes(element_bytes),
                      .width = width,
                  },
          });
    }
    stencil_shape_detail::Insert(
        ranked, count,
        stencil_shape_detail::RankedCandidate{
            .shape = direct,
            .score =
                {
                    .direct = true,
                    // Direct candidates all perform the same N(2r+1) reads. Do
                    // not materialize that potentially overflowing common term.
                    .global_reads = 0u,
                    .launched_lanes = groups * width,
                    .groups = groups,
                    .shared_bytes = 0u,
                    .width = width,
                },
        });
  }
  result.count = count;
  for (std::size_t index = 0u; index < count; ++index) {
    result.values[index] = ranked[index].shape;
  }
  return result;
}

enum class StencilGpuCandidateDecision : std::uint8_t {
  Skip,
  Select,
  Abort,
};

struct StencilGpuShapeSelection final {
  StencilGpuShape shape{};
  bool aborted{};
};

template <typename Decide>
[[nodiscard]] constexpr StencilGpuShapeSelection SelectStencilGpuShapeCandidate(
    const StencilGpuShapeCandidates &candidates,
    Decide &&decide) noexcept(noexcept(decide(StencilGpuShape{}))) {
  for (const StencilGpuShape shape : candidates) {
    const StencilGpuCandidateDecision decision = decide(shape);
    if (decision == StencilGpuCandidateDecision::Select) {
      return StencilGpuShapeSelection{.shape = shape};
    }
    if (decision == StencilGpuCandidateDecision::Abort) {
      return StencilGpuShapeSelection{.aborted = true};
    }
  }
  return {};
}

[[nodiscard]] constexpr StencilGpuShape
SelectStencilGpuShape(const rund::kernel::u64 element_count,
                      const rund::kernel::u64 radius,
                      const rund::kernel::u32 element_bytes,
                      const StencilGpuCapabilities capabilities) noexcept {
  const StencilGpuShapeCandidates candidates =
      RankStencilGpuShapes(element_count, radius, element_bytes, capabilities);
  return candidates.empty() ? StencilGpuShape{} : candidates[0u];
}

inline constexpr StencilGpuShape kStencilMaximumSourceShape =
    StencilGpuShape::shared(256u, 256u);

[[nodiscard]] constexpr bool
StencilPhysicalGroupsFit(const rund::kernel::u64 element_count,
                         const rund::kernel::u64 maximum_groups,
                         const StencilGpuShape shape) noexcept {
  const rund::kernel::u64 groups =
      StencilPhysicalGroupCount(element_count, shape);
  return groups != 0u && groups <= maximum_groups &&
         groups <= std::numeric_limits<rund::kernel::u32>::max();
}

[[nodiscard]] constexpr bool
StencilVulkanDispatchFits(const rund::kernel::u64 element_count,
                          const rund::kernel::u64 maximum_groups,
                          const StencilGpuShape shape) noexcept {
  return element_count <= std::numeric_limits<rund::kernel::u32>::max() &&
         StencilPhysicalGroupsFit(element_count, maximum_groups, shape);
}

[[nodiscard]] constexpr rund::kernel::u64
StencilDirectGlobalReads(const rund::kernel::u32 active_lanes,
                         const rund::kernel::u32 radius) noexcept {
  return static_cast<rund::kernel::u64>(active_lanes) *
         (2u * static_cast<rund::kernel::u64>(radius) + 1u);
}

[[nodiscard]] constexpr rund::kernel::u32
StencilLeftHaloDistinctInputs(const rund::kernel::u64 group_base,
                              const rund::kernel::u32 radius) noexcept {
  return static_cast<rund::kernel::u32>(std::min<rund::kernel::u64>(
      group_base, static_cast<rund::kernel::u64>(radius)));
}

[[nodiscard]] constexpr rund::kernel::u32
StencilRightHaloDistinctInputs(const rund::kernel::u64 element_count,
                               const rund::kernel::u64 group_base,
                               const rund::kernel::u32 active_lanes,
                               const rund::kernel::u32 radius) noexcept {
  const rund::kernel::u64 remaining =
      group_base >= element_count ? 0u : element_count - group_base;
  const rund::kernel::u64 covered = std::min<rund::kernel::u64>(
      remaining, static_cast<rund::kernel::u64>(active_lanes));
  const rund::kernel::u64 available = remaining - covered;
  return static_cast<rund::kernel::u32>(std::min<rund::kernel::u64>(
      available, static_cast<rund::kernel::u64>(radius)));
}

[[nodiscard]] constexpr rund::kernel::u64
StencilSharedGlobalReads(const rund::kernel::u64 element_count,
                         const rund::kernel::u64 group_base,
                         const rund::kernel::u32 active_lanes,
                         const rund::kernel::u32 radius) noexcept {
  return static_cast<rund::kernel::u64>(active_lanes) +
         StencilLeftHaloDistinctInputs(group_base, radius) +
         StencilRightHaloDistinctInputs(element_count, group_base, active_lanes,
                                        radius);
}

[[nodiscard]] bool StencilShapeOk(const rund::kernel::StencilDesc &desc,
                                  const rund::kernel::StencilPlan &plan,
                                  const StencilBinds &bindings) noexcept;

} // namespace rund::node::accel::detail
