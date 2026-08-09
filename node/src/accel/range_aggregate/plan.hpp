#pragma once

#include "model.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

namespace rund::node::accel::detail {
namespace range_plan_detail {

inline constexpr rund::kernel::u128 kU128Maximum =
    ~static_cast<rund::kernel::u128>(0u);

[[nodiscard]] constexpr bool Add(const rund::kernel::u128 left,
                                 const rund::kernel::u128 right,
                                 rund::kernel::u128 &out) noexcept {
  if (left > kU128Maximum - right) {
    return false;
  }
  out = left + right;
  return true;
}

[[nodiscard]] constexpr bool Multiply(const rund::kernel::u128 left,
                                      const rund::kernel::u128 right,
                                      rund::kernel::u128 &out) noexcept {
  if (right != 0u && left > kU128Maximum / right) {
    return false;
  }
  out = left * right;
  return true;
}

[[nodiscard]] constexpr bool
Accumulate(rund::kernel::u128 &target,
           const rund::kernel::u128 value) noexcept {
  return Add(target, value, target);
}

[[nodiscard]] constexpr bool
AccumulateProduct(rund::kernel::u128 &target, const rund::kernel::u128 left,
                  const rund::kernel::u128 right) noexcept {
  rund::kernel::u128 product = 0u;
  return Multiply(left, right, product) && Accumulate(target, product);
}

[[nodiscard]] constexpr bool
AccumulateBytes(rund::kernel::u128 &target, const rund::kernel::u128 elements,
                const rund::kernel::u32 element_bytes) noexcept {
  return AccumulateProduct(target, elements, element_bytes);
}

[[nodiscard]] constexpr rund::kernel::u64
Groups(const rund::kernel::u64 count, const rund::kernel::u32 width) noexcept {
  return width == 0u ? 0u
                     : count / width +
                           static_cast<rund::kernel::u64>(count % width != 0u);
}

[[nodiscard]] constexpr bool
FitsGroups(const RangeCaps &capabilities,
           const rund::kernel::u64 groups) noexcept {
  return groups != 0u && (capabilities.cpu_only() ||
                          groups <= capabilities.maximum_group_count());
}

[[nodiscard]] constexpr rund::kernel::u32
SharedRadiusCapacity(const RangeCaps &capabilities,
                     const rund::kernel::u32 width,
                     const rund::kernel::u32 element_bytes) noexcept {
  const rund::kernel::u32 occupancy =
      capabilities.shared_memory_occupancy_budget();
  if (occupancy == 0u || element_bytes == 0u) {
    return 0u;
  }
  const rund::kernel::u64 element_capacity =
      capabilities.shared_memory_limit() / occupancy / element_bytes;
  if (element_capacity <= width) {
    return 0u;
  }
  const rund::kernel::u64 radius_capacity = (element_capacity - width) / 2u;
  return static_cast<rund::kernel::u32>(
      std::min<rund::kernel::u64>(width, radius_capacity));
}

[[nodiscard]] constexpr bool
SharedBudgetFits(const RangeCaps &capabilities,
                 const rund::kernel::u64 shared_bytes) noexcept {
  const rund::kernel::u32 occupancy =
      capabilities.shared_memory_occupancy_budget();
  return occupancy != 0u &&
         shared_bytes <= capabilities.shared_memory_limit() / occupancy;
}

struct CandidateEvaluation final {
  explicit constexpr CandidateEvaluation(
      const RangeCandidate candidate_value) noexcept
      : candidate(candidate_value) {}

  RangeCandidate candidate;
  RangeCost cost{};
  std::array<RangeStagePlan, kRangeStageCap> stages{};
  std::size_t stage_count{};
  std::array<RangeTempReq, kRangeTempCap> temporaries{};
  std::size_t temporary_count{};
};

[[nodiscard]] constexpr bool
AppendStage(CandidateEvaluation &evaluation, const RangeCaps &capabilities,
            const RangeStageKind disposition, const std::uint8_t level,
            const rund::kernel::u64 element_count,
            const rund::kernel::u64 groups, const rund::kernel::u32 width,
            bool &overflow) noexcept {
  if (evaluation.stage_count == evaluation.stages.size() ||
      !FitsGroups(capabilities, groups)) {
    return false;
  }
  evaluation.stages[evaluation.stage_count++] = RangeStagePlan{
      .disposition = disposition,
      .level = level,
      .element_count = element_count,
      .groups = groups,
      .width = width,
  };
  if (evaluation.cost.dispatch_count ==
      std::numeric_limits<rund::kernel::u64>::max()) {
    overflow = true;
    return false;
  }
  ++evaluation.cost.dispatch_count;
  if (width != 0u &&
      !AccumulateProduct(evaluation.cost.launched_lanes, groups, width)) {
    overflow = true;
    return false;
  }
  return true;
}

[[nodiscard]] constexpr bool
AppendTemporary(CandidateEvaluation &evaluation, const RangeTempRole role,
                const std::uint8_t ordinal, const rund::kernel::u64 bytes,
                const rund::kernel::u64 alignment,
                const std::uint8_t first_stage, const std::uint8_t last_stage,
                bool &overflow) noexcept {
  if (evaluation.temporary_count == evaluation.temporaries.size()) {
    return false;
  }
  rund::kernel::u64 scratch = 0u;
  if (!rund::kernel::checked::add(evaluation.cost.scratch_bytes, bytes,
                                  scratch)) {
    overflow = true;
    return false;
  }
  evaluation.cost.scratch_bytes = scratch;
  evaluation.temporaries[evaluation.temporary_count++] =
      RangeTempReq{.role = role,
                   .ordinal = ordinal,
                   .bytes = bytes,
                   .alignment = alignment,
                   .first_stage = first_stage,
                   .last_stage = last_stage};
  return true;
}

[[nodiscard]] constexpr std::optional<CandidateEvaluation>
BuildDirect(const RangeShape &shape, const RangeCaps &capabilities,
            const RangeCandidate candidate, bool &overflow) noexcept {
  CandidateEvaluation evaluation{candidate};
  rund::kernel::u128 twice_radius = 0u;
  rund::kernel::u128 window = 0u;
  rund::kernel::u128 read_elements = 0u;
  if (!Multiply(shape.radius(), 2u, twice_radius) ||
      !Add(twice_radius, 1u, window) ||
      !Multiply(shape.element_count(), window, read_elements) ||
      !AccumulateBytes(evaluation.cost.global_read_bytes, read_elements,
                       shape.element_bytes()) ||
      !AccumulateBytes(evaluation.cost.global_write_bytes,
                       shape.element_count(), shape.element_bytes()) ||
      !Multiply(shape.element_count(), twice_radius,
                evaluation.cost.combine_ops)) {
    overflow = true;
    return std::nullopt;
  }

  const bool cpu = candidate.width() == 0u;
  const rund::kernel::u64 groups =
      cpu ? 1u : Groups(shape.element_count(), candidate.width());
  if (!AppendStage(evaluation, capabilities, RangeStageKind::Direct, 0u,
                   shape.element_count(), groups, candidate.width(),
                   overflow)) {
    return std::nullopt;
  }
  if (cpu) {
    evaluation.cost.launched_lanes = shape.element_count();
  }
  return evaluation;
}

[[nodiscard]] constexpr std::optional<CandidateEvaluation>
BuildSharedHalo(const RangeShape &shape, const RangeCaps &capabilities,
                const RangeCandidate candidate, bool &overflow) noexcept {
  if (candidate.radius_capacity() < shape.radius()) {
    return std::nullopt;
  }
  CandidateEvaluation evaluation{candidate};
  const rund::kernel::u64 groups =
      Groups(shape.element_count(), candidate.width());
  if (!FitsGroups(capabilities, groups)) {
    return std::nullopt;
  }
  const rund::kernel::u64 shared_elements =
      candidate.width() + 2u * candidate.radius_capacity();
  const rund::kernel::u64 shared_bytes =
      shared_elements * shape.element_bytes();
  if (!SharedBudgetFits(capabilities, shared_bytes)) {
    return std::nullopt;
  }
  evaluation.cost.shared_bytes = shared_bytes;

  rund::kernel::u128 read_elements = shape.element_count();
  if (groups > 1u) {
    rund::kernel::u128 group_term = 0u;
    rund::kernel::u128 halo = 0u;
    const rund::kernel::u64 tail =
        shape.element_count() - (groups - 1u) * candidate.width();
    if (!Multiply(groups, 2u, group_term) || group_term < 3u ||
        !Multiply(group_term - 3u, shape.radius(), halo) ||
        !Add(halo, std::min<rund::kernel::u64>(shape.radius(), tail), halo) ||
        !Add(read_elements, halo, read_elements)) {
      overflow = true;
      return std::nullopt;
    }
  }
  rund::kernel::u128 twice_radius = 0u;
  if (!AccumulateBytes(evaluation.cost.global_read_bytes, read_elements,
                       shape.element_bytes()) ||
      !AccumulateBytes(evaluation.cost.global_write_bytes,
                       shape.element_count(), shape.element_bytes()) ||
      !Multiply(shape.radius(), 2u, twice_radius) ||
      !Multiply(shape.element_count(), twice_radius,
                evaluation.cost.combine_ops) ||
      !AppendStage(evaluation, capabilities, RangeStageKind::SharedHalo, 0u,
                   shape.element_count(), groups, candidate.width(),
                   overflow)) {
    overflow = true;
    return std::nullopt;
  }
  return evaluation;
}

[[nodiscard]] constexpr std::optional<CandidateEvaluation>
BuildPrefixDifference(const RangeShape &shape, const RangeCaps &capabilities,
                      const RangeCandidate candidate, bool &overflow) noexcept {
  if (!shape.traits().associative() || !shape.traits().has_identity() ||
      !shape.traits().invertible()) {
    return std::nullopt;
  }
  CandidateEvaluation evaluation{candidate};
  const rund::kernel::u64 local_shared =
      static_cast<rund::kernel::u64>(candidate.width()) * shape.element_bytes();
  if (!SharedBudgetFits(capabilities, local_shared)) {
    return std::nullopt;
  }
  evaluation.cost.shared_bytes = local_shared;
  const RangePrefixExec hierarchy = PlanRangePrefixTree(
      shape.element_count(), candidate.width(), shape.element_bytes(),
      capabilities.maximum_group_count());
  if (!hierarchy.ok() || hierarchy.stage_count() == 0u ||
      hierarchy.stage_count() >= std::numeric_limits<std::uint8_t>::max()) {
    return std::nullopt;
  }
  if (!AppendTemporary(evaluation, RangeTempRole::PrefixValues, 0u,
                       shape.payload_bytes(), shape.element_bytes(), 0u,
                       static_cast<std::uint8_t>(hierarchy.stage_count()),
                       overflow)) {
    return std::nullopt;
  }

  for (std::size_t index = 0u; index < hierarchy.stage_count(); ++index) {
    const RangeStagePlan stage = hierarchy.stage(index);
    if (!AppendStage(evaluation, capabilities, stage.disposition, stage.level,
                     stage.element_count, stage.groups, stage.width,
                     overflow)) {
      return std::nullopt;
    }
    if (stage.disposition == RangeStageKind::PrefixBlock ||
        stage.disposition == RangeStageKind::PrefixSummary) {
      if (!AccumulateBytes(evaluation.cost.global_read_bytes,
                           stage.element_count, shape.element_bytes()) ||
          !AccumulateBytes(evaluation.cost.global_write_bytes,
                           stage.element_count, shape.element_bytes()) ||
          !AccumulateProduct(evaluation.cost.combine_ops, stage.groups,
                             2u * (candidate.width() - 1u))) {
        overflow = true;
        return std::nullopt;
      }
      if (stage.groups > 1u) {
        bool found = false;
        for (std::size_t temporary_index = 0u;
             temporary_index < hierarchy.temporary_count(); ++temporary_index) {
          const RangeTempReq temporary = hierarchy.temporary(temporary_index);
          if (temporary.role != RangeTempRole::BlockSummaries ||
              temporary.ordinal != stage.level) {
            continue;
          }
          found = AppendTemporary(evaluation, temporary.role, temporary.ordinal,
                                  temporary.bytes, temporary.alignment,
                                  temporary.first_stage, temporary.last_stage,
                                  overflow) &&
                  AccumulateBytes(evaluation.cost.global_write_bytes,
                                  stage.groups, shape.element_bytes());
          break;
        }
        if (!found) {
          overflow = true;
          return std::nullopt;
        }
      }
      continue;
    }
    if (stage.disposition != RangeStageKind::PrefixFixup) {
      overflow = true;
      return std::nullopt;
    }
    const rund::kernel::u64 adjusted =
        stage.element_count -
        std::min<rund::kernel::u64>(stage.element_count, candidate.width());
    if (!AccumulateBytes(evaluation.cost.global_read_bytes,
                         static_cast<rund::kernel::u128>(adjusted) * 2u,
                         shape.element_bytes()) ||
        !AccumulateBytes(evaluation.cost.global_write_bytes, adjusted,
                         shape.element_bytes()) ||
        !Accumulate(evaluation.cost.combine_ops, adjusted)) {
      overflow = true;
      return std::nullopt;
    }
  }

  const rund::kernel::u64 output_groups =
      Groups(shape.element_count(), candidate.width());
  const rund::kernel::u64 left_prefix_reads =
      shape.radius() < shape.element_count()
          ? shape.element_count() - shape.radius() - 1u
          : 0u;
  rund::kernel::u128 output_reads = 0u;
  rund::kernel::u128 endpoint_reads = 0u;
  if (!Multiply(shape.radius(), 2u, endpoint_reads) ||
      !Add(shape.element_count(), left_prefix_reads, output_reads) ||
      !Add(output_reads, endpoint_reads, output_reads) ||
      !AppendStage(evaluation, capabilities, RangeStageKind::PrefixWindow, 0u,
                   shape.element_count(), output_groups, candidate.width(),
                   overflow) ||
      !AccumulateBytes(evaluation.cost.global_read_bytes, output_reads,
                       shape.element_bytes()) ||
      !AccumulateBytes(evaluation.cost.global_write_bytes,
                       shape.element_count(), shape.element_bytes()) ||
      !Accumulate(evaluation.cost.inverse_ops, left_prefix_reads) ||
      !Accumulate(evaluation.cost.scale_ops, endpoint_reads) ||
      !Accumulate(evaluation.cost.combine_ops, endpoint_reads)) {
    overflow = true;
    return std::nullopt;
  }
  return evaluation;
}

[[nodiscard]] constexpr std::optional<CandidateEvaluation>
BuildBlockPrefixSuffix(const RangeShape &shape, const RangeCaps &capabilities,
                       const RangeCandidate candidate,
                       bool &overflow) noexcept {
  if (!shape.traits().associative() || !shape.traits().has_identity() ||
      !shape.traits().idempotent() || !shape.traits().ordered()) {
    return std::nullopt;
  }
  rund::kernel::u64 twice_radius = 0u;
  rund::kernel::u64 window = 0u;
  rund::kernel::u64 padded = 0u;
  if (!rund::kernel::checked::mul(shape.radius(), 2u, twice_radius) ||
      !rund::kernel::checked::add(twice_radius, 1u, window) ||
      !rund::kernel::checked::add(shape.element_count(), twice_radius,
                                  padded)) {
    overflow = true;
    return std::nullopt;
  }
  const rund::kernel::u64 blocks =
      padded / window + static_cast<rund::kernel::u64>(padded % window != 0u);
  const rund::kernel::u64 prepare_groups = Groups(blocks, candidate.width());
  const rund::kernel::u64 output_groups =
      Groups(shape.element_count(), candidate.width());
  if (!FitsGroups(capabilities, prepare_groups) ||
      !FitsGroups(capabilities, output_groups)) {
    return std::nullopt;
  }
  rund::kernel::u64 value_bytes = 0u;
  if (!rund::kernel::checked::mul(padded, shape.element_bytes(), value_bytes)) {
    overflow = true;
    return std::nullopt;
  }

  CandidateEvaluation evaluation{candidate};
  if (!AppendTemporary(evaluation, RangeTempRole::ForwardValues, 0u,
                       value_bytes, shape.element_bytes(), 0u, 1u, overflow) ||
      !AppendTemporary(evaluation, RangeTempRole::BackwardValues, 0u,
                       value_bytes, shape.element_bytes(), 0u, 1u, overflow) ||
      !AppendStage(evaluation, capabilities, RangeStageKind::BlockPrefixSuffix,
                   0u, padded, prepare_groups, candidate.width(), overflow) ||
      !AppendStage(evaluation, capabilities, RangeStageKind::BlockWindow, 0u,
                   shape.element_count(), output_groups, candidate.width(),
                   overflow) ||
      !AccumulateBytes(evaluation.cost.global_read_bytes,
                       static_cast<rund::kernel::u128>(padded) * 2u,
                       shape.element_bytes()) ||
      !AccumulateBytes(evaluation.cost.global_read_bytes,
                       static_cast<rund::kernel::u128>(shape.element_count()) *
                           2u,
                       shape.element_bytes()) ||
      !AccumulateBytes(evaluation.cost.global_write_bytes,
                       static_cast<rund::kernel::u128>(padded) * 2u,
                       shape.element_bytes()) ||
      !AccumulateBytes(evaluation.cost.global_write_bytes,
                       shape.element_count(), shape.element_bytes()) ||
      !AccumulateProduct(evaluation.cost.combine_ops, padded - blocks, 2u) ||
      !Accumulate(evaluation.cost.combine_ops, shape.element_count())) {
    overflow = true;
    return std::nullopt;
  }
  return evaluation;
}

[[nodiscard]] constexpr bool LessOrEqual(const RangeCost &left,
                                         const RangeCost &right) noexcept {
  return left.global_read_bytes <= right.global_read_bytes &&
         left.global_write_bytes <= right.global_write_bytes &&
         left.combine_ops <= right.combine_ops &&
         left.inverse_ops <= right.inverse_ops &&
         left.scale_ops <= right.scale_ops &&
         left.shared_bytes <= right.shared_bytes &&
         left.scratch_bytes <= right.scratch_bytes &&
         left.dispatch_count <= right.dispatch_count &&
         left.launched_lanes <= right.launched_lanes;
}

[[nodiscard]] constexpr bool Dominates(const RangeCost &left,
                                       const RangeCost &right) noexcept {
  return LessOrEqual(left, right) && !(left == right);
}

template <typename T>
[[nodiscard]] constexpr int CompareScalar(const T left,
                                          const T right) noexcept {
  return left < right ? -1 : (right < left ? 1 : 0);
}

[[nodiscard]] constexpr int
CompareLexicographic(const CandidateEvaluation &left,
                     const CandidateEvaluation &right) noexcept {
#define RUND_RANGE_COMPARE(field)                                              \
  if (const int order = CompareScalar(left.cost.field, right.cost.field);      \
      order != 0) {                                                            \
    return order;                                                              \
  }
  RUND_RANGE_COMPARE(global_read_bytes)
  RUND_RANGE_COMPARE(global_write_bytes)
  RUND_RANGE_COMPARE(combine_ops)
  RUND_RANGE_COMPARE(inverse_ops)
  RUND_RANGE_COMPARE(scale_ops)
  RUND_RANGE_COMPARE(scratch_bytes)
  RUND_RANGE_COMPARE(dispatch_count)
  RUND_RANGE_COMPARE(shared_bytes)
  RUND_RANGE_COMPARE(launched_lanes)
#undef RUND_RANGE_COMPARE
  if (const int order = CompareScalar(
          static_cast<std::uint8_t>(left.candidate.disposition()),
          static_cast<std::uint8_t>(right.candidate.disposition()));
      order != 0) {
    return order;
  }
  if (const int order =
          CompareScalar(left.candidate.width(), right.candidate.width());
      order != 0) {
    return order;
  }
  return CompareScalar(left.candidate.radius_capacity(),
                       right.candidate.radius_capacity());
}

class IdentityBuilder final {
public:
  constexpr IdentityBuilder() noexcept = default;

  template <typename T>
    requires(std::is_integral_v<T> && sizeof(T) <= sizeof(std::uint64_t))
  constexpr void add(const T value) noexcept {
    add64(static_cast<std::uint64_t>(value));
  }

  constexpr void add(const rund::kernel::u128 value) noexcept {
    add64(static_cast<std::uint64_t>(value >> 64u));
    add64(static_cast<std::uint64_t>(value));
  }

  [[nodiscard]] constexpr RangeIdentity finish() const noexcept {
    return RangeIdentity{.hi = hi_, .lo = lo_};
  }

private:
  constexpr void add64(const std::uint64_t value) noexcept {
    hi_ ^= value + 0x9e3779b97f4a7c15ull + (hi_ << 6u) + (hi_ >> 2u);
    hi_ *= 0xbf58476d1ce4e5b9ull;
    lo_ ^= value + 0x94d049bb133111ebull + (lo_ << 7u) + (lo_ >> 3u);
    lo_ *= 0x9e3779b185ebca87ull;
  }
  std::uint64_t hi_{0x6a09e667f3bcc909ull};
  std::uint64_t lo_{0xbb67ae8584caa73bull};
};

[[nodiscard]] constexpr RangeIdentity
SourceIdentity(const RangeShape &shape, const RangeCaps &capabilities,
               const RangeCandidate candidate) noexcept {
  IdentityBuilder identity{};
  identity.add(0x72616e67652d7372ull); // "range-sr"
  identity.add(1u);
  identity.add(static_cast<std::uint8_t>(capabilities.source_variant()));
  identity.add(static_cast<std::uint8_t>(candidate.disposition()));
  identity.add(candidate.width());
  identity.add(candidate.radius_capacity());
  identity.add(static_cast<std::uint8_t>(shape.traits().operation()));
  identity.add(static_cast<std::uint8_t>(shape.traits().domain()));
  identity.add(static_cast<std::uint8_t>(shape.traits().arithmetic_law()));
  identity.add(static_cast<std::uint8_t>(shape.boundary()));
  identity.add(shape.element_bytes());
  return identity.finish();
}

[[nodiscard]] constexpr RangeIdentity
ExecutionIdentity(const RangeShape &shape,
                  const CandidateEvaluation &evaluation,
                  const RangeIdentity source_identity) noexcept {
  IdentityBuilder identity{};
  identity.add(0x72616e67652d6578ull); // "range-ex"
  identity.add(source_identity.hi);
  identity.add(source_identity.lo);
  identity.add(shape.element_count());
  identity.add(shape.radius());
  identity.add(evaluation.stage_count);
  for (std::size_t index = 0u; index < evaluation.stage_count; ++index) {
    const RangeStagePlan &stage = evaluation.stages[index];
    identity.add(static_cast<std::uint8_t>(stage.disposition));
    identity.add(stage.level);
    identity.add(stage.element_count);
    identity.add(stage.groups);
    identity.add(stage.width);
  }
  identity.add(evaluation.temporary_count);
  for (std::size_t index = 0u; index < evaluation.temporary_count; ++index) {
    const RangeTempReq &temporary = evaluation.temporaries[index];
    identity.add(static_cast<std::uint8_t>(temporary.role));
    identity.add(temporary.ordinal);
    identity.add(temporary.bytes);
    identity.add(temporary.alignment);
    identity.add(temporary.first_stage);
    identity.add(temporary.last_stage);
  }
  identity.add(evaluation.cost.global_read_bytes);
  identity.add(evaluation.cost.global_write_bytes);
  identity.add(evaluation.cost.combine_ops);
  identity.add(evaluation.cost.inverse_ops);
  identity.add(evaluation.cost.scale_ops);
  identity.add(evaluation.cost.shared_bytes);
  identity.add(evaluation.cost.scratch_bytes);
  identity.add(evaluation.cost.dispatch_count);
  identity.add(evaluation.cost.launched_lanes);
  return identity.finish();
}

} // namespace range_plan_detail

[[nodiscard]] constexpr RangePlan
PlanRange(const RangeShape &shape, const RangeCaps &capabilities) noexcept {
  using namespace range_plan_detail;
  if (!shape.valid()) {
    return RangePlan::rejected("compute_range_aggregate_shape_invalid");
  }
  if (!capabilities.valid()) {
    return RangePlan::rejected(
        capabilities.available()
            ? "compute_range_aggregate_capabilities_invalid"
            : "compute_range_aggregate_unavailable");
  }

  std::array<std::optional<CandidateEvaluation>, kRangeCandidateCap>
      evaluations{};
  std::size_t evaluation_count = 0u;
  bool saw_overflow = false;
  const auto append = [&](std::optional<CandidateEvaluation> evaluation) {
    if (evaluation.has_value() && evaluation_count < evaluations.size()) {
      evaluations[evaluation_count++] = std::move(evaluation);
    }
  };

  if (capabilities.cpu_only()) {
    append(BuildDirect(shape, capabilities, RangeCandidate::direct_cpu(),
                       saw_overflow));
  } else {
    for (const rund::kernel::u32 width : kRangeWidths) {
      if (!capabilities.supports_width(width)) {
        continue;
      }
      if (capabilities.supports(RangeSupport::Direct)) {
        const std::optional<RangeCandidate> candidate =
            RangeCandidate::direct_gpu(width);
        if (candidate.has_value()) {
          append(BuildDirect(shape, capabilities, *candidate, saw_overflow));
        }
      }
      if (capabilities.supports(RangeSupport::SharedHalo)) {
        const rund::kernel::u32 capacity =
            SharedRadiusCapacity(capabilities, width, shape.element_bytes());
        const std::optional<RangeCandidate> candidate =
            RangeCandidate::shared_halo(width, capacity);
        if (candidate.has_value()) {
          append(
              BuildSharedHalo(shape, capabilities, *candidate, saw_overflow));
        }
      }
      if (capabilities.supports(RangeSupport::PrefixDifference) &&
          shape.traits().invertible()) {
        const std::optional<RangeCandidate> candidate =
            RangeCandidate::prefix_difference(width);
        if (candidate.has_value()) {
          append(BuildPrefixDifference(shape, capabilities, *candidate,
                                       saw_overflow));
        }
      }
      if (capabilities.supports(RangeSupport::BlockPrefixSuffix) &&
          shape.traits().idempotent() && shape.traits().ordered()) {
        const std::optional<RangeCandidate> candidate =
            RangeCandidate::block_prefix_suffix(width);
        if (candidate.has_value()) {
          append(BuildBlockPrefixSuffix(shape, capabilities, *candidate,
                                        saw_overflow));
        }
      }
    }
  }

  if (evaluation_count == 0u) {
    return RangePlan::rejected(
        saw_overflow ? "compute_range_aggregate_cost_overflow"
                     : "compute_range_aggregate_candidate_unavailable");
  }

  std::array<bool, kRangeCandidateCap> pareto{};
  std::size_t pareto_count = 0u;
  for (std::size_t index = 0u; index < evaluation_count; ++index) {
    bool dominated = false;
    for (std::size_t other = 0u; other < evaluation_count; ++other) {
      if (other != index &&
          Dominates(evaluations[other]->cost, evaluations[index]->cost)) {
        dominated = true;
        break;
      }
    }
    pareto[index] = !dominated;
    pareto_count += static_cast<std::size_t>(!dominated);
  }

  std::size_t selected = evaluation_count;
  for (std::size_t index = 0u; index < evaluation_count; ++index) {
    if (pareto[index] && (selected == evaluation_count ||
                          CompareLexicographic(*evaluations[index],
                                               *evaluations[selected]) < 0)) {
      selected = index;
    }
  }
  if (selected == evaluation_count || evaluation_count > 255u ||
      pareto_count > 255u) {
    return RangePlan::rejected("compute_range_aggregate_candidate_unavailable");
  }

  const CandidateEvaluation &choice = *evaluations[selected];
  const RangeIdentity source_identity =
      SourceIdentity(shape, capabilities, choice.candidate);
  const RangeIdentity execution_identity =
      ExecutionIdentity(shape, choice, source_identity);
  return RangePlan::selected(shape, choice.candidate, choice.cost,
                             choice.stage_count, choice.temporary_count,
                             static_cast<std::uint8_t>(evaluation_count),
                             static_cast<std::uint8_t>(pareto_count),
                             source_identity, execution_identity);
}

static_assert(std::is_nothrow_move_constructible_v<RangePlan>);
static_assert(std::is_nothrow_move_assignable_v<RangePlan>);

} // namespace rund::node::accel::detail
