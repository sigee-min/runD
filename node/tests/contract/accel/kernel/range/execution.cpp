#include "local.hpp"

#include "src/accel/context/internal/execution.hpp"
#include "src/accel/metal/range/local.hpp"
#include "src/accel/source/hash.hpp"
#include "src/accel/vulkan/kernel/manifest.hpp"
#include "src/accel/vulkan/kernel/ops/table.hpp"
#include "src/accel/vulkan/kernel/pipeline/source.hpp"
#include "src/accel/vulkan/kernel/pipeline/template.hpp"
#include "src/accel/vulkan/range/local.hpp"
#include "src/accel/window/shape.hpp"

#include <kernel/program/compute/window/plan.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

namespace node_accel_contract::range {
namespace {

using rund::kernel::ComputeDomain;
using rund::kernel::u32;
using rund::kernel::u64;
using namespace rund::node::accel::detail;

[[nodiscard]] constexpr bool PrefixHierarchyContract() {
  constexpr std::uint8_t direct_prefix =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::PrefixDifference);
  const RangeCaps capabilities =
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
          std::numeric_limits<u32>::max(), direct_prefix);
  const RangePlan plan =
      ContractPlanRange(Shape(RangeOp::Sum, 4097u, 4097u), capabilities);
  constexpr std::array expected{
      RangeStageKind::PrefixBlock,   RangeStageKind::PrefixSummary,
      RangeStageKind::PrefixSummary, RangeStageKind::PrefixFixup,
      RangeStageKind::PrefixWindow,
  };
  if (!plan.ok() ||
      plan.candidate().disposition() != RangePath::PrefixDifference ||
      plan.stage_count() != expected.size() || plan.temporary_count() != 3u ||
      plan.cost().scratch_bytes != 16656u ||
      plan.cost().dispatch_count != expected.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < expected.size(); ++index) {
    if (plan.stage(index).disposition != expected[index]) {
      return false;
    }
  }
  return plan.temporary(0u).role == RangeTempRole::PrefixValues &&
         plan.temporary(0u).first_stage == 0u &&
         plan.temporary(0u).last_stage == 4u &&
         plan.temporary(1u).role == RangeTempRole::BlockSummaries &&
         plan.temporary(1u).bytes == 65u * 4u &&
         plan.temporary(1u).last_stage == 4u &&
         plan.temporary(2u).bytes == 2u * 4u &&
         plan.temporary(2u).last_stage == 3u;
}

[[nodiscard]] constexpr bool PrefixStageSubstrateContract() {
  constexpr auto hierarchy =
      PlanRangePrefixTree(4097u, 64u, 4u, std::numeric_limits<u32>::max());
  constexpr auto flat_single = PlanRangeFlatPrefix(256u, 1u, 128u, 4u);
  constexpr auto flat_multiple = PlanRangeFlatPrefix(513u, 3u, 128u, 8u);
  constexpr auto invalid = PlanRangeFlatPrefix(0u, 0u, 128u, 4u);
  constexpr auto dispatch_limited = PlanRangePrefixTree(4097u, 64u, 4u, 64u);
  return hierarchy.ok() &&
         hierarchy.disposition() == RangePrefixKind::Hierarchical &&
         hierarchy.stage_count() == 5u && hierarchy.temporary_count() == 2u &&
         hierarchy.stage(0u) ==
             RangeStagePlan{.disposition = RangeStageKind::PrefixBlock,
                            .level = 0u,
                            .element_count = 4097u,
                            .groups = 65u,
                            .width = 64u} &&
         hierarchy.stage(4u) ==
             RangeStagePlan{.disposition = RangeStageKind::PrefixFixup,
                            .level = 0u,
                            .element_count = 4097u,
                            .groups = 65u,
                            .width = 64u} &&
         hierarchy.temporary(0u).bytes == 260u &&
         hierarchy.temporary(0u).last_stage == 4u &&
         hierarchy.temporary(1u).bytes == 8u &&
         hierarchy.temporary(1u).last_stage == 3u && flat_single.ok() &&
         flat_single.disposition() == RangePrefixKind::FlatBlockTotals &&
         flat_single.stage_count() == 1u &&
         flat_single.temporary_count() == 1u &&
         flat_single.stage(0u).groups == 1u &&
         flat_single.temporary(0u).bytes == 4u &&
         flat_single.temporary(0u).last_stage == 0u && flat_multiple.ok() &&
         flat_multiple.disposition() == RangePrefixKind::FlatBlockTotals &&
         flat_multiple.stage_count() == 3u &&
         flat_multiple.stage(0u).groups == 3u &&
         flat_multiple.stage(1u).groups == 1u &&
         flat_multiple.stage(2u).groups == 3u &&
         flat_multiple.temporary(0u).bytes == 24u &&
         flat_multiple.temporary(0u).last_stage == 2u && !invalid.ok() &&
         !dispatch_limited.ok();
}

[[nodiscard]] constexpr bool ResidentRunContract() {
  constexpr std::uint8_t direct_prefix =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::PrefixDifference);
  constexpr std::uint8_t direct_block =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::BlockPrefixSuffix);
  const RangeShape resident_sum =
      AffineShape(RangeOp::Sum, RangeBoundary::Clamp, 4097u, 4097u, 8195u, 1u,
                  4097u, 4u, ComputeDomain::U32, RangeCount::U32);
  const RangeShape resident_min =
      AffineShape(RangeOp::Minimum, RangeBoundary::Clip, 4097u, 4097u, 8195u,
                  1u, 4097u, 4u, ComputeDomain::I32, RangeCount::U64);
  const RangePlan prefix = ContractPlanRange(
      resident_sum, Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
                        std::numeric_limits<u32>::max(), direct_prefix));
  const RangePlan block = ContractPlanRange(
      resident_min, Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
                        std::numeric_limits<u32>::max(), direct_block));
  if (!prefix.ok() || !block.ok() || prefix.stage_count() != 5u ||
      block.stage_count() != 2u || RangeRun::make(prefix, 4098u).has_value()) {
    return false;
  }

  const auto empty = RangeRun::make(prefix, 0u);
  const auto one = RangeRun::make(prefix, 1u);
  const auto middle = RangeRun::make(prefix, 65u);
  const auto block_middle = RangeRun::make(block, 65u);
  if (!empty.has_value() || !one.has_value() || !middle.has_value() ||
      !block_middle.has_value()) {
    return false;
  }
  constexpr std::array<u64, 5u> middle_groups{2u, 1u, 0u, 0u, 2u};
  for (std::size_t index = 0u; index < prefix.stage_count(); ++index) {
    const auto empty_stage = empty->stage(index);
    const auto one_stage = one->stage(index);
    const auto middle_stage = middle->stage(index);
    if (!empty_stage.has_value() || !one_stage.has_value() ||
        !middle_stage.has_value() || empty_stage->groups() != 0u ||
        middle_stage->groups() != middle_groups[index] ||
        middle_stage->params().input_count() != 65u ||
        middle_stage->params().output_count() != 65u) {
      return false;
    }
  }
  if (one->stage(0u)->groups() != 1u || one->stage(1u)->groups() != 0u ||
      one->stage(3u)->groups() != 0u || one->stage(4u)->groups() != 1u) {
    return false;
  }
  const auto prepare = block_middle->stage(0u);
  const auto query = block_middle->stage(1u);
  return prepare.has_value() && query.has_value() && prepare->groups() == 1u &&
         prepare->params().stage_element_count() == 8259u &&
         prepare->params().stage_aux_count() == 2u && query->groups() == 2u &&
         query->params().stage_element_count() == 65u;
}

[[nodiscard]] constexpr bool BlockPrefixSuffixContract() {
  constexpr std::uint8_t direct_block =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::BlockPrefixSuffix);
  const RangeCaps capabilities =
      Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
          std::numeric_limits<u32>::max(), direct_block);
  const RangePlan plan =
      ContractPlanRange(Shape(RangeOp::Minimum, 4097u, 4097u), capabilities);
  return plan.ok() &&
         plan.candidate().disposition() == RangePath::BlockPrefixSuffix &&
         plan.stage_count() == 2u && plan.temporary_count() == 2u &&
         plan.stage(0u) ==
             RangeStagePlan{.disposition = RangeStageKind::BlockPrefixSuffix,
                            .level = 0u,
                            .element_count = 12291u,
                            .groups = 1u,
                            .width = 64u} &&
         plan.stage(1u).disposition == RangeStageKind::BlockWindow &&
         plan.stage(1u).groups == 65u && plan.cost().scratch_bytes == 98328u &&
         plan.cost().dispatch_count == 2u;
}

[[nodiscard]] constexpr bool MetalBlockGeometryContract() {
  constexpr auto support = RangeSupportBit(RangeSupport::Direct) |
                           RangeSupportBit(RangeSupport::BlockPrefixSuffix);
  const auto shape =
      AffineShape(RangeOp::Minimum, RangeBoundary::Clip, 4097u, 4097u, 8195u,
                  1u, 4097u, 4u, ComputeDomain::I32, RangeCount::U64);
  const auto caps = Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
                        std::numeric_limits<u32>::max(), support);
  const auto plan = ContractPlanRange(shape, caps);
  if (!plan.ok() ||
      plan.candidate().disposition() != RangePath::BlockPrefixSuffix ||
      plan.stage(0u).groups != 2u || plan.stage(1u).groups != 1u ||
      plan.temporary_count() != 1u ||
      plan.cost().scratch_bytes != 12291u * 4u ||
      plan.cost().shared_bytes != 0u) {
    return false;
  }
  for (const u64 count : {0u, 1u, 65u, 4097u}) {
    const auto run = RangeRun::make(plan, count);
    if (!run.has_value()) {
      return false;
    }
    const auto query = run->stage(1u);
    if (!query || query->groups() != (count == 0u ? 0u : 1u)) {
      return false;
    }
    const auto stage = run->stage(0u);
    const u64 span = count == 0u ? 0u : count - 1u + 8195u;
    const u64 expected = span / 8195u + (span % 8195u != 0u);
    if (!stage.has_value() || stage->groups() != expected ||
        stage->params().stage_aux_count() != expected) {
      return false;
    }
  }
  const auto without_shared = Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u,
                                  0u, std::numeric_limits<u32>::max(), support);
  const auto direct = ContractPlanRange(shape, without_shared);
  return direct.ok() &&
         direct.candidate().disposition() == RangePath::BlockPrefixSuffix;
}

[[nodiscard]] bool ExecutionProjectionContract() {
  constexpr std::uint8_t direct_shared =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::SharedHalo);
  constexpr std::uint8_t direct_prefix =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::PrefixDifference);
  constexpr std::uint8_t direct_block =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::BlockPrefixSuffix);
  const RangeCaps shared_caps =
      Gpu(RangeSource::Metal, kRangeWidth128Bit, 128u, 4u,
          (128u + 2u * 9u) * 4u * 4u, std::numeric_limits<u32>::max(),
          direct_shared);
  const RangeCaps prefix_caps =
      Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 4u, 32768u,
          std::numeric_limits<u32>::max(), direct_prefix);
  const RangeCaps block_caps =
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
          std::numeric_limits<u32>::max(), direct_block);
  const RangePlan shared =
      ContractPlanRange(Shape(RangeOp::Sum, 129u, 9u), shared_caps);
  const RangePlan prefix =
      ContractPlanRange(Shape(RangeOp::Sum, 4097u, 4097u), prefix_caps);
  const RangePlan block =
      ContractPlanRange(Shape(RangeOp::Maximum, 4097u, 4097u), block_caps);
  const RangePlan cpu = ContractPlanRange(Shape(RangeOp::Sum, 17u, 3u),
                                          RangeCaps::cpu_reference());
  const auto shared_exec = RangeExec::from(shared);
  const auto prefix_exec = RangeExec::from(prefix);
  const auto block_exec = RangeExec::from(block);
  if (!shared_exec.has_value() || !prefix_exec.has_value() ||
      !block_exec.has_value() || RangeExec::from(cpu).has_value()) {
    return false;
  }
  const auto shared_params = shared_exec->stage_params(0u);
  const auto prefix_scratch = prefix_exec->stage_scratch(0u);
  const auto block_scratch = block_exec->stage_scratch(0u);
  return shared_exec->width() == 128u &&
         shared_exec->shared_radius_capacity() == 9u &&
         shared_exec->uses_shared_halo() && prefix_exec->width() == 64u &&
         prefix_exec->shared_radius_capacity() == 0u &&
         !prefix_exec->uses_shared_halo() && block_exec->width() == 64u &&
         block_exec->shared_radius_capacity() == 0u &&
         !block_exec->uses_shared_halo() &&
         shared_exec->descriptor_count() == 3u &&
         prefix_exec->descriptor_count() == 5u &&
         block_exec->descriptor_count() == 5u &&
         shared_exec->static_shared_bytes() == (128u + 18u) * 4u &&
         prefix_exec->static_shared_bytes() == 64u * 4u &&
         block_exec->static_shared_bytes() == 0u && shared_params.has_value() &&
         shared_params->stage() ==
             static_cast<u32>(RangeStageKind::SharedHalo) &&
         !shared_exec->stage_scratch(0u)->uses_global_scratch() &&
         prefix_scratch.has_value() && prefix_scratch->uses_global_scratch() &&
         prefix_scratch->first().role() == RangeTempRole::PrefixValues &&
         block_scratch.has_value() && block_scratch->uses_global_scratch() &&
         block_scratch->first().role() == RangeTempRole::ForwardValues &&
         block_scratch->second().role() == RangeTempRole::ForwardValues &&
         shared_exec->source_identity() == shared.source_identity() &&
         prefix_exec->execution_identity() == prefix.execution_identity();
}

[[nodiscard]] constexpr bool AffineCostAndTopologyContract() {
  constexpr std::uint8_t direct = RangeSupportBit(RangeSupport::Direct);
  constexpr std::uint8_t direct_shared =
      direct | RangeSupportBit(RangeSupport::SharedHalo);
  constexpr std::uint8_t direct_prefix =
      direct | RangeSupportBit(RangeSupport::PrefixDifference);
  constexpr std::uint8_t direct_block =
      direct | RangeSupportBit(RangeSupport::BlockPrefixSuffix);
  const RangeCaps direct_caps =
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u,
          std::numeric_limits<u32>::max(), direct);
  const RangeShape clamp =
      AffineShape(RangeOp::Sum, RangeBoundary::Clamp, 5u, 3u, 3u, 2u, 1u);
  const RangeShape clip =
      AffineShape(RangeOp::Sum, RangeBoundary::Clip, 5u, 3u, 3u, 2u, 1u);
  const RangePlan clamp_direct = ContractPlanRange(clamp, direct_caps);
  const RangePlan clip_direct = ContractPlanRange(clip, direct_caps);
  const RangeShape prefix_shape =
      AffineShape(RangeOp::Sum, RangeBoundary::Clip, 257u, 65u, 129u, 2u, 64u);
  const RangePlan prefix = ContractPlanRange(
      prefix_shape, Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
                        std::numeric_limits<u32>::max(), direct_prefix));
  const RangeShape block_shape = AffineShape(
      RangeOp::Minimum, RangeBoundary::Clip, 101u, 7u, 50u, 3u, 10u);
  const RangeCaps block_caps =
      Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
          std::numeric_limits<u32>::max(), direct_block);
  const RangePlan block = ContractPlanRange(block_shape, block_caps);
  const RangeShape block_clamp_shape = AffineShape(
      RangeOp::Minimum, RangeBoundary::Clamp, 257u, 65u, 129u, 2u, 64u);
  const RangeShape block_clip_shape = AffineShape(
      RangeOp::Minimum, RangeBoundary::Clip, 257u, 65u, 129u, 2u, 64u);
  const RangePlan block_clamp =
      ContractPlanRange(block_clamp_shape, block_caps);
  const RangePlan block_clip = ContractPlanRange(block_clip_shape, block_caps);
  const RangePlan shared_for_affine = ContractPlanRange(
      AffineShape(RangeOp::Sum, RangeBoundary::Clamp, 65u, 32u, 3u, 2u, 1u),
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
          std::numeric_limits<u32>::max(), direct_shared));
  return clamp_direct.ok() && clip_direct.ok() &&
         clamp_direct.cost().global_read_bytes == 9u * 4u &&
         clamp_direct.cost().global_write_bytes == 3u * 4u &&
         clamp_direct.cost().combine_ops == 6u &&
         clip_direct.cost().global_read_bytes == 7u * 4u &&
         clip_direct.cost().global_write_bytes == 3u * 4u &&
         clip_direct.cost().combine_ops == 4u && prefix.ok() &&
         prefix.candidate().disposition() == RangePath::PrefixDifference &&
         prefix.stage(prefix.stage_count() - 1u).element_count == 65u &&
         prefix.stage(prefix.stage_count() - 1u).groups == 2u &&
         prefix.temporary(0u).bytes == 257u * 4u && block.ok() &&
         block.candidate().disposition() == RangePath::BlockPrefixSuffix &&
         *block_shape.affine_span() == 68u &&
         block.stage(0u).element_count == 68u &&
         block.stage(1u).element_count == 7u &&
         block.cost().scratch_bytes == 2u * 68u * 4u && block_clamp.ok() &&
         block_clip.ok() &&
         block_clamp.candidate().disposition() ==
             RangePath::BlockPrefixSuffix &&
         block_clip.candidate() == block_clamp.candidate() &&
         block_clamp.cost().global_read_bytes == (2u * 257u + 2u * 65u) * 4u &&
         block_clip.cost().global_read_bytes ==
             (2u * (257u - 64u) + 2u * 65u) * 4u &&
         shared_for_affine.ok() &&
         shared_for_affine.candidate().disposition() != RangePath::SharedHalo;
}

[[nodiscard]] constexpr bool AffinePaddedDirectContract() {
  constexpr std::uint8_t direct = RangeSupportBit(RangeSupport::Direct);
  const RangeShape clamp =
      AffineShape(RangeOp::Sum, RangeBoundary::Clamp, 3u, 2u, 5u, 5u, 4u);
  const RangeShape clip =
      AffineShape(RangeOp::Sum, RangeBoundary::Clip, 3u, 2u, 5u, 5u, 4u);
  const RangePlan clamp_plan = ContractPlanRange(
      clamp, Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u,
                 std::numeric_limits<u32>::max(), direct));
  const RangePlan clip_plan = ContractPlanRange(
      clip, Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
                std::numeric_limits<u32>::max(), direct));
  constexpr std::array<u32, 3u> input{2u, 5u, 7u};
  constexpr std::array<u32, 2u> clamp_expected{10u, 33u};
  constexpr std::array<u32, 2u> clip_expected{2u, 12u};
  std::array<u32, 2u> clamp_actual{};
  std::array<u32, 2u> clip_actual{};
  for (u64 output = 0u; output < 2u; ++output) {
    const u64 anchor = output * 5u;
    for (u64 slot = 0u; slot < 5u; ++slot) {
      u64 index = 0u;
      bool valid = true;
      if (slot < 4u) {
        const u64 delta = 4u - slot;
        if (anchor < delta) {
          index = 0u;
        } else {
          index = anchor - delta;
          if (index >= input.size()) {
            index = input.size() - 1u;
          }
        }
      } else if (anchor >= input.size() || slot - 4u >= input.size() - anchor) {
        index = input.size() - 1u;
      } else {
        index = anchor + slot - 4u;
      }
      clamp_actual[output] += input[index];

      if (slot < 4u) {
        const u64 delta = 4u - slot;
        if (anchor < delta) {
          valid = false;
        } else {
          index = anchor - delta;
          valid = index < input.size();
        }
      } else if (anchor >= input.size() || slot - 4u >= input.size() - anchor) {
        valid = false;
      } else {
        index = anchor + slot - 4u;
      }
      if (valid) {
        clip_actual[output] += input[index];
      }
    }
  }
  return clamp_plan.ok() && clip_plan.ok() &&
         clamp_plan.candidate().disposition() == RangePath::Direct &&
         clip_plan.candidate().disposition() == RangePath::Direct &&
         clamp_plan.cost().global_read_bytes == 10u * 4u &&
         clip_plan.cost().global_read_bytes == 3u * 4u &&
         clamp_actual == clamp_expected && clip_actual == clip_expected;
}

[[nodiscard]] bool PrefixQueryCostContract() {
  for (auto n : {1u, 63u, 64u, 65u, 257u, 4097u})
    for (auto p : {0u, 1u, 64u, 128u})
      for (auto stride : {1u, 3u, 67u})
        for (auto boundary : {RangeBoundary::Clip, RangeBoundary::Clamp}) {
          const unsigned k = 2 * n + p + 513, q = (n + p - 1) / stride + 1;
          auto shape = AffineShape(RangeOp::Sum, boundary, n, q, k, stride, p);
          auto caps = Gpu(RangeSource::Metal, kRangeWidth64Bit, 64, 4, 32768,
                          UINT32_MAX,
                          RangeSupportBit(RangeSupport::PrefixDifference) |
                              RangeSupportBit(RangeSupport::Direct));
          bool overflow = false;
          auto evaluated = range_plan_detail::BuildPrefixDifference(
              shape, caps, *RangeCandidate::prefix_difference(64), overflow);
          if (!evaluated || overflow)
            return false;
          const auto &plan = *evaluated;
          unsigned long long reads = 0, writes = 0, combine = 0, inverse = 0;
          for (size_t s = 0; s + 1 < plan.stage_count; ++s) {
            auto stage = plan.stages[s];
            if (stage.disposition == RangeStageKind::PrefixFixup) {
              auto adjusted = stage.element_count -
                              std::min<uint64_t>(stage.element_count, 64);
              reads += 2 * adjusted;
              writes += adjusted;
              combine += adjusted;
            } else {
              reads += stage.element_count;
              writes += stage.element_count;
              if (stage.groups > 1)
                writes += stage.groups;
              combine += stage.groups * 126;
            }
          }
          for (unsigned j = 0; j < q; ++j) {
            auto anchor = uint64_t(j) * stride;
            auto left = anchor < p ? 0 : anchor - p;
            auto right = std::min<uint64_t>(n - 1, anchor + k - p - 1);
            reads++;
            writes++;
            if (right / 64) {
              reads++;
              combine++;
            }
            if (left) {
              reads++;
              inverse++;
              if ((left - 1) / 64) {
                reads++;
                inverse++;
              }
            }
            if (boundary == RangeBoundary::Clamp) {
              if (anchor < p) {
                reads++;
                combine++;
              }
              if (anchor + k - p > n) {
                reads++;
                combine++;
              }
            }
          }
          if (plan.cost.global_read_bytes != reads * 4 ||
              plan.cost.global_write_bytes != writes * 4 ||
              plan.cost.combine_ops != combine ||
              plan.cost.inverse_ops != inverse) {
            return false;
          }
        }
  return true;
}

[[nodiscard]] bool TiledDifferenceContract() {
  for (const auto source : {RangeSource::Metal, RangeSource::Vulkan}) {
    const auto caps = Gpu(source);
    const auto small =
        ContractPlanRange(Shape(RangeOp::Sum, 262144u, 4u), caps);
    const auto large =
        ContractPlanRange(Shape(RangeOp::Sum, 262144u, 1024u), caps);
    const auto huge =
        ContractPlanRange(Shape(RangeOp::Sum, 262144u, 262144u), caps);
    if (!small.ok() || !large.ok() || !huge.ok() ||
        small.candidate().disposition() != RangePath::SharedHalo ||
        large.candidate().disposition() != RangePath::TiledDifference ||
        huge.candidate().disposition() != RangePath::PrefixDifference ||
        large.stage_count() != 1u || large.temporary_count() != 0u ||
        large.cost().scratch_bytes != 0u) {
      return false;
    }
    const auto resident = ContractPlanRange(
        AffineShape(RangeOp::Sum, RangeBoundary::Clamp, 262144u, 262144u, 2049u,
                    1u, 1024u, 4u, ComputeDomain::U32, RangeCount::U32),
        caps);
    const auto exec = RangeExec::from(resident);
    if (!exec || exec->uses_global_scratch() ||
        exec->descriptor_count() != 3u ||
        exec->stage_scratch(0u)->uses_global_scratch()) {
      return false;
    }
    for (const u64 active : {0u, 1u, 4095u, 4096u, 4097u, 262144u}) {
      const auto run = RangeRun::make(resident, active);
      const auto stage = run ? run->stage(0u) : std::nullopt;
      const u64 tile = exec->width() * kRangeTileOutputsPerLane;
      if (!stage || stage->groups() != active / tile + (active % tile != 0u)) {
        return false;
      }
    }
    if (RangeRun::make(resident, 262145u)) {
      return false;
    }
    // Count actual indexed reads for every tile's anchor and recurrence.
    u64 reads = 0u, groups = 0u;
    const u64 tile = large.candidate().width() * kRangeTileOutputsPerLane;
    for (u64 base = 0u; base < 262144u; base += tile) {
      ++groups;
      const u64 left = base < 1024u ? 0u : base - 1024u;
      const u64 right = std::min<u64>(262143u, base + 1024u);
      reads += right - left + 1u + (base < 1024u) + (base + 1024u >= 262144u);
    }
    reads += 2u * (262144u - groups);
    if (large.cost().global_read_bytes != reads * 4u ||
        large.cost().global_write_bytes != 262144u * 4u) {
      return false;
    }
  }
  return true;
}

static_assert(PrefixHierarchyContract());
static_assert(PrefixStageSubstrateContract());
static_assert(ResidentRunContract());
static_assert(BlockPrefixSuffixContract());
static_assert(MetalBlockGeometryContract());
static_assert(AffineCostAndTopologyContract());
static_assert(AffinePaddedDirectContract());

} // namespace

bool ExecutionContract() {
  return PrefixHierarchyContract() && PrefixStageSubstrateContract() &&
         ResidentRunContract() && BlockPrefixSuffixContract() &&
         MetalBlockGeometryContract() && ExecutionProjectionContract() &&
         AffineCostAndTopologyContract() && AffinePaddedDirectContract() &&
         PrefixQueryCostContract() && TiledDifferenceContract();
}

} // namespace node_accel_contract::range
