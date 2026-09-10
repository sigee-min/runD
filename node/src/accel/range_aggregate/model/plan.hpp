#pragma once

#include "block.hpp"
#include "candidate.hpp"
#include "capability.hpp"
#include "prefix.hpp"
#include "shape.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace rund::node::accel::detail {

class RangePlan final {
public:
  RangePlan() = delete;

  [[nodiscard]] static constexpr RangePlan
  rejected(const char *const reason) noexcept {
    return RangePlan{RangePlanKind::Rejected, reason};
  }

  [[nodiscard]] constexpr RangePlanKind disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr bool ok() const noexcept {
    return disposition_ == RangePlanKind::Selected;
  }

  [[nodiscard]] constexpr const char *reason() const noexcept {
    return reason_;
  }

  [[nodiscard]] constexpr const RangeShape &shape() const noexcept {
    return selection().shape;
  }

  [[nodiscard]] constexpr const RangeCandidate &candidate() const noexcept {
    return selection().candidate;
  }

  [[nodiscard]] constexpr RangeSource source_variant() const noexcept {
    return selection().source_variant;
  }

  [[nodiscard]] constexpr const RangeCost &cost() const noexcept {
    return selection().cost;
  }

  [[nodiscard]] constexpr std::size_t stage_count() const noexcept {
    return selection().stage_count;
  }

  [[nodiscard]] constexpr RangeStagePlan
  stage(const std::size_t index) const noexcept {
    assert(index < stage_count());
    const Selection &selected = selection();
    const RangePath disposition = selected.candidate.disposition();
    if (disposition == RangePath::Direct) {
      return RangeStagePlan{.disposition = RangeStageKind::Direct,
                            .level = 0u,
                            .element_count = selected.shape.output_count(),
                            .groups =
                                selected.candidate.width() == 0u
                                    ? 1u
                                    : Groups(selected.shape.output_count(),
                                             selected.candidate.width()),
                            .width = selected.candidate.width()};
    }
    if (disposition == RangePath::TiledDifference) {
      return RangeStagePlan{.disposition = RangeStageKind::TiledDifference,
                            .level = 0u,
                            .element_count = selected.shape.output_count(),
                            .groups = Groups(selected.shape.output_count(),
                                             selected.candidate.width() *
                                                 kRangeTileOutputsPerLane),
                            .width = selected.candidate.width()};
    }
    if (disposition == RangePath::SharedHalo) {
      return RangeStagePlan{.disposition = RangeStageKind::SharedHalo,
                            .level = 0u,
                            .element_count = selected.shape.input_count(),
                            .groups = Groups(selected.shape.input_count(),
                                             selected.candidate.width()),
                            .width = selected.candidate.width()};
    }
    if (disposition == RangePath::BlockPrefixSuffix) {
      if (selected.source_variant == RangeSource::Cpu) {
        const std::optional<rund::kernel::u64> span =
            selected.shape.affine_span();
        assert(span.has_value());
        return index == 0u
                   ? RangeStagePlan{.disposition =
                                        RangeStageKind::BlockPrefixSuffix,
                                    .level = 0u,
                                    .element_count = *span,
                                    .groups = 1u,
                                    .width = 0u}
                   : RangeStagePlan{.disposition = RangeStageKind::BlockWindow,
                                    .level = 0u,
                                    .element_count =
                                        selected.shape.output_count(),
                                    .groups = 1u,
                                    .width = 0u};
      }
      if (index == 0u) {
        const std::optional<rund::kernel::u64> span =
            selected.shape.affine_span();
        assert(span.has_value());
        const rund::kernel::u64 padded = *span;
        const rund::kernel::u64 window = selected.shape.window_size();
        const rund::kernel::u64 blocks = Groups(padded, window);
        return RangeStagePlan{
            .disposition = RangeStageKind::BlockPrefixSuffix,
            .level = 0u,
            .element_count = padded,
            .groups = RangeBlockGroups(selected.source_variant, blocks,
                                       selected.candidate.width()),
            .width = selected.candidate.width()};
      }
      return RangeStagePlan{
          .disposition = RangeStageKind::BlockWindow,
          .level = 0u,
          .element_count = selected.shape.output_count(),
          .groups = RangeBlockQueryGroups(
              selected.source_variant, selected.shape.output_count(),
              selected.shape.stride(), selected.shape.window_size(),
              selected.candidate.width()),
          .width = selected.candidate.width()};
    }

    if (selected.source_variant == RangeSource::Cpu) {
      return index == 0u
                 ? RangeStagePlan{.disposition =
                                      RangeStageKind::PrefixSequential,
                                  .level = 0u,
                                  .element_count = selected.shape.input_count(),
                                  .groups = 1u,
                                  .width = 0u}
                 : RangeStagePlan{.disposition = RangeStageKind::PrefixWindow,
                                  .level = 0u,
                                  .element_count =
                                      selected.shape.output_count(),
                                  .groups = 1u,
                                  .width = 0u};
    }
    const RangePrefixExec prefix = PlanRangePrefixTree(
        selected.shape.input_count(), selected.candidate.width(),
        selected.shape.element_bytes(),
        std::numeric_limits<rund::kernel::u64>::max());
    assert(prefix.ok() &&
           selected.stage_count ==
               prefix.stage_count() + 1u -
                   static_cast<std::size_t>(selected.shape.input_count() >
                                            selected.candidate.width()));
    if (index + 1u < selected.stage_count) {
      return prefix.stage(index);
    }
    return RangeStagePlan{.disposition = RangeStageKind::PrefixWindow,
                          .level = 0u,
                          .element_count = selected.shape.output_count(),
                          .groups = Groups(selected.shape.output_count(),
                                           selected.candidate.width()),
                          .width = selected.candidate.width()};
  }

  [[nodiscard]] constexpr std::size_t temporary_count() const noexcept {
    return selection().temporary_count;
  }

  [[nodiscard]] constexpr RangeTempReq
  temporary(const std::size_t index) const noexcept {
    assert(index < temporary_count());
    const Selection &selected = selection();
    if (selected.candidate.disposition() == RangePath::PrefixDifference) {
      if (index == 0u) {
        return RangeTempReq{
            .role = RangeTempRole::PrefixValues,
            .ordinal = 0u,
            .bytes = selected.shape.payload_bytes(),
            .alignment = selected.shape.element_bytes(),
            .first_stage = 0u,
            .last_stage = static_cast<std::uint8_t>(selected.stage_count - 1u)};
      }
      assert(selected.source_variant != RangeSource::Cpu);
      const RangePrefixExec prefix = PlanRangePrefixTree(
          selected.shape.input_count(), selected.candidate.width(),
          selected.shape.element_bytes(),
          std::numeric_limits<rund::kernel::u64>::max());
      assert(prefix.ok() && index - 1u < prefix.temporary_count() &&
             selected.temporary_count == prefix.temporary_count() + 1u);
      return prefix.temporary(index - 1u);
    }
    const std::optional<rund::kernel::u64> span = selected.shape.affine_span();
    assert(span.has_value());
    const rund::kernel::u64 bytes = *span * selected.shape.element_bytes();
    return RangeTempReq{.role = index == 0u ? RangeTempRole::ForwardValues
                                            : RangeTempRole::BackwardValues,
                        .ordinal = 0u,
                        .bytes = bytes,
                        .alignment = selected.shape.element_bytes(),
                        .first_stage = 0u,
                        .last_stage = 1u};
  }

  [[nodiscard]] constexpr std::uint8_t legal_candidate_count() const noexcept {
    return selection().legal_candidate_count;
  }

  [[nodiscard]] constexpr std::uint8_t pareto_candidate_count() const noexcept {
    return selection().pareto_candidate_count;
  }

  [[nodiscard]] constexpr RangeIdentity source_identity() const noexcept {
    return selection().source_identity;
  }

  [[nodiscard]] constexpr RangeIdentity execution_identity() const noexcept {
    return selection().execution_identity;
  }

private:
  friend constexpr RangePlan BuildRangePlan(const RangeShape &,
                                            const RangeCaps &) noexcept;

  [[nodiscard]] static constexpr RangePlan
  selected(const RangeShape shape, const RangeCandidate candidate,
           const RangeSource source_variant, const RangeCost cost,
           const std::size_t stage_count, const std::size_t temporary_count,
           const std::uint8_t legal_candidate_count,
           const std::uint8_t pareto_candidate_count,
           const RangeIdentity source_identity,
           const RangeIdentity execution_identity) noexcept {
    return RangePlan{shape,
                     candidate,
                     source_variant,
                     cost,
                     stage_count,
                     temporary_count,
                     legal_candidate_count,
                     pareto_candidate_count,
                     source_identity,
                     execution_identity};
  }

  struct Selection final {
    RangeShape shape;
    RangeCandidate candidate;
    RangeSource source_variant{};
    RangeCost cost{};
    std::size_t stage_count{};
    std::size_t temporary_count{};
    std::uint8_t legal_candidate_count{};
    std::uint8_t pareto_candidate_count{};
    RangeIdentity source_identity{};
    RangeIdentity execution_identity{};
  };

  [[nodiscard]] constexpr const Selection &selection() const noexcept {
    assert(disposition_ == RangePlanKind::Selected && selection_.has_value());
    return *selection_;
  }

  [[nodiscard]] static constexpr rund::kernel::u64
  Groups(const rund::kernel::u64 count,
         const rund::kernel::u64 width) noexcept {
    return width == 0u
               ? 0u
               : count / width +
                     static_cast<rund::kernel::u64>(count % width != 0u);
  }

  constexpr RangePlan(const RangePlanKind disposition,
                      const char *const reason) noexcept
      : disposition_(disposition), reason_(reason) {}

  constexpr RangePlan(const RangeShape shape, const RangeCandidate candidate,
                      const RangeSource source_variant, const RangeCost cost,
                      const std::size_t stage_count,
                      const std::size_t temporary_count,
                      const std::uint8_t legal_candidate_count,
                      const std::uint8_t pareto_candidate_count,
                      const RangeIdentity source_identity,
                      const RangeIdentity execution_identity) noexcept
      : disposition_(RangePlanKind::Selected),
        selection_(Selection{shape, candidate, source_variant, cost,
                             stage_count, temporary_count,
                             legal_candidate_count, pareto_candidate_count,
                             source_identity, execution_identity}),
        reason_("ok") {}

  RangePlanKind disposition_;
  std::optional<Selection> selection_{};
  const char *reason_{};
};

static_assert(std::is_trivially_copyable_v<RangeIdentity>);
static_assert(std::is_trivially_copyable_v<RangeCost>);
static_assert(std::is_trivially_copyable_v<RangeTempReq>);
static_assert(std::is_trivially_copyable_v<RangeStagePlan>);
static_assert(std::is_trivially_copyable_v<RangePrefixExec>);

} // namespace rund::node::accel::detail
