#pragma once

#include "candidate.hpp"

#include <kernel/core/checked.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace rund::node::accel::detail {

// This derives physical prefix stages and temporary lifetimes. It does not
// choose an aggregate algorithm: Range selects PrefixDifference, and
// native Scan projects its own observable-prefix semantics into the flat form.
enum class RangePrefixKind : std::uint8_t {
  Rejected,
  Hierarchical,
  FlatBlockTotals,
};

namespace range_prefix_detail {
class Builder;
}

class RangePrefixExec final {
public:
  RangePrefixExec() = delete;

  [[nodiscard]] constexpr bool ok() const noexcept {
    return disposition_ != RangePrefixKind::Rejected;
  }

  [[nodiscard]] constexpr RangePrefixKind disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr const char *reason() const noexcept {
    return reason_;
  }

  [[nodiscard]] constexpr rund::kernel::u32 width() const noexcept {
    return selection().width;
  }

  [[nodiscard]] constexpr std::size_t stage_count() const noexcept {
    return selection().stage_count;
  }

  [[nodiscard]] constexpr RangeStagePlan
  stage(const std::size_t index) const noexcept {
    assert(index < stage_count());
    return selection().stages[index];
  }

  [[nodiscard]] constexpr std::size_t temporary_count() const noexcept {
    return selection().temporary_count;
  }

  [[nodiscard]] constexpr RangeTempReq
  temporary(const std::size_t index) const noexcept {
    assert(index < temporary_count());
    return selection().temporaries[index];
  }

private:
  struct Selection final {
    rund::kernel::u32 width{};
    std::array<RangeStagePlan, kRangeStageCap> stages{};
    std::size_t stage_count{};
    std::array<RangeTempReq, kRangeTempCap> temporaries{};
    std::size_t temporary_count{};
  };

  friend class range_prefix_detail::Builder;

  [[nodiscard]] static constexpr RangePrefixExec
  rejected(const char *const reason) noexcept {
    return RangePrefixExec{RangePrefixKind::Rejected, reason};
  }

  [[nodiscard]] static constexpr RangePrefixExec
  selected(const RangePrefixKind disposition,
           const Selection selection) noexcept {
    return RangePrefixExec{disposition, selection};
  }

  [[nodiscard]] constexpr const Selection &selection() const noexcept {
    assert(ok() && selection_.has_value());
    return *selection_;
  }

  constexpr RangePrefixExec(const RangePrefixKind disposition,
                            const char *const reason) noexcept
      : disposition_(disposition), reason_(reason) {}

  constexpr RangePrefixExec(const RangePrefixKind disposition,
                            const Selection selection) noexcept
      : disposition_(disposition), selection_(selection), reason_("ok") {}

  RangePrefixKind disposition_;
  std::optional<Selection> selection_{};
  const char *reason_;
};

namespace range_prefix_detail {

class Builder final {
public:
  constexpr explicit Builder(const rund::kernel::u32 width) noexcept
      : selection_{.width = width} {}

  [[nodiscard]] static constexpr RangePrefixExec rejected() noexcept {
    return RangePrefixExec::rejected("compute_range_aggregate_prefix_invalid");
  }

  [[nodiscard]] constexpr bool
  append_stage(const RangeStageKind disposition, const std::uint8_t level,
               const rund::kernel::u64 element_count,
               const rund::kernel::u64 groups,
               const rund::kernel::u32 width) noexcept {
    if (selection_.stage_count == selection_.stages.size() || groups == 0u) {
      return false;
    }
    selection_.stages[selection_.stage_count++] = RangeStagePlan{
        .disposition = disposition,
        .level = level,
        .element_count = element_count,
        .groups = groups,
        .width = width,
    };
    return true;
  }

  [[nodiscard]] constexpr bool append_temporary(
      const RangeTempRole role, const std::uint8_t ordinal,
      const rund::kernel::u64 bytes, const rund::kernel::u32 alignment,
      const std::uint8_t first_stage, const std::uint8_t last_stage) noexcept {
    if (selection_.temporary_count == selection_.temporaries.size() ||
        bytes == 0u || alignment == 0u || first_stage > last_stage) {
      return false;
    }
    selection_.temporaries[selection_.temporary_count++] =
        RangeTempReq{.role = role,
                     .ordinal = ordinal,
                     .bytes = bytes,
                     .alignment = alignment,
                     .first_stage = first_stage,
                     .last_stage = last_stage};
    return true;
  }

  [[nodiscard]] constexpr std::size_t temporary_count() const noexcept {
    return selection_.temporary_count;
  }

  [[nodiscard]] constexpr RangeStagePlan
  stage(const std::size_t index) const noexcept {
    assert(index < selection_.stage_count);
    return selection_.stages[index];
  }

  constexpr void set_last_stage(const std::size_t index,
                                const std::uint8_t last_stage) noexcept {
    assert(index < selection_.temporary_count);
    selection_.temporaries[index].last_stage = last_stage;
  }

  [[nodiscard]] constexpr RangePrefixExec
  finish(const RangePrefixKind disposition) const noexcept {
    return RangePrefixExec::selected(disposition, selection_);
  }

private:
  RangePrefixExec::Selection selection_;
};

[[nodiscard]] constexpr bool
valid_width(const rund::kernel::u32 width) noexcept {
  return width == 64u || width == 128u || width == 256u;
}

[[nodiscard]] constexpr rund::kernel::u64
groups(const rund::kernel::u64 count, const rund::kernel::u32 width) noexcept {
  return width == 0u
             ? 0u
             : count / width +
                   static_cast<rund::kernel::u64>(count % width != 0u);
}

} // namespace range_prefix_detail

// Derives the work-efficient recursive prefix hierarchy used by the
// PrefixDifference candidate. Every hierarchy stage must fit the physical
// single-stage dispatch limit supplied by the backend capability projection.
[[nodiscard]] constexpr RangePrefixExec
PlanRangePrefixTree(const rund::kernel::u64 element_count,
                    const rund::kernel::u32 width,
                    const rund::kernel::u32 element_bytes,
                    const rund::kernel::u64 maximum_group_count) noexcept {
  using namespace range_prefix_detail;
  if (element_count == 0u || !valid_width(width) ||
      (element_bytes != 4u && element_bytes != 8u) ||
      maximum_group_count == 0u) {
    return Builder::rejected();
  }

  Builder builder{width};
  std::array<std::size_t, kRangeTempCap> summary_index{};
  std::size_t level_count = 0u;
  rund::kernel::u64 values = element_count;
  while (true) {
    if (level_count == summary_index.size()) {
      return Builder::rejected();
    }
    const rund::kernel::u64 group_count = groups(values, width);
    if (group_count == 0u || group_count > maximum_group_count ||
        !builder.append_stage(level_count == 0u ? RangeStageKind::PrefixBlock
                                                : RangeStageKind::PrefixSummary,
                              static_cast<std::uint8_t>(level_count), values,
                              group_count, width)) {
      return Builder::rejected();
    }
    ++level_count;
    if (group_count == 1u) {
      break;
    }
    rund::kernel::u64 bytes = 0u;
    if (!rund::kernel::checked::mul(group_count, element_bytes, bytes) ||
        !builder.append_temporary(
            RangeTempRole::BlockSummaries,
            static_cast<std::uint8_t>(level_count - 1u), bytes, element_bytes,
            static_cast<std::uint8_t>(level_count - 1u),
            static_cast<std::uint8_t>(level_count - 1u))) {
      return Builder::rejected();
    }
    summary_index[level_count - 1u] = builder.temporary_count() - 1u;
    values = group_count;
  }

  for (std::size_t level = level_count - 1u; level != 0u; --level) {
    const std::size_t child = level - 1u;
    const RangeStagePlan child_stage = builder.stage(child);
    if (!builder.append_stage(
            RangeStageKind::PrefixFixup, static_cast<std::uint8_t>(child),
            child_stage.element_count, child_stage.groups, width)) {
      return Builder::rejected();
    }
    builder.set_last_stage(
        summary_index[child],
        static_cast<std::uint8_t>(2u * level_count - 2u - child));
  }
  return builder.finish(RangePrefixKind::Hierarchical);
}

// Derives the native Scan physical graph: block-local prefixes, one flat
// block-total prefix, then final offset/materialization. The caller supplies
// the semantic block count; device dispatch chunking remains a backend command
// concern, so this records full logical block count rather than an
// adapter-specific chunk count.
[[nodiscard]] constexpr RangePrefixExec
PlanRangeFlatPrefix(const rund::kernel::u64 element_count,
                    const rund::kernel::u64 block_count,
                    const rund::kernel::u32 width,
                    const rund::kernel::u32 element_bytes) noexcept {
  using namespace range_prefix_detail;
  if (element_count == 0u || block_count == 0u || !valid_width(width) ||
      (element_bytes != 4u && element_bytes != 8u)) {
    return Builder::rejected();
  }
  rund::kernel::u64 summary_bytes = 0u;
  if (block_count == 0u ||
      !rund::kernel::checked::mul(block_count, element_bytes, summary_bytes)) {
    return Builder::rejected();
  }

  Builder builder{width};
  if (!builder.append_stage(RangeStageKind::PrefixBlock, 0u, element_count,
                            block_count, width) ||
      !builder.append_temporary(
          RangeTempRole::BlockSummaries, 0u, summary_bytes, element_bytes, 0u,
          static_cast<std::uint8_t>(block_count == 1u ? 0u : 2u))) {
    return Builder::rejected();
  }
  if (block_count == 1u) {
    return builder.finish(RangePrefixKind::FlatBlockTotals);
  }
  if (!builder.append_stage(RangeStageKind::PrefixSummary, 0u, block_count, 1u,
                            width) ||
      !builder.append_stage(RangeStageKind::PrefixFixup, 0u, element_count,
                            block_count, width)) {
    return Builder::rejected();
  }
  return builder.finish(RangePrefixKind::FlatBlockTotals);
}

} // namespace rund::node::accel::detail
