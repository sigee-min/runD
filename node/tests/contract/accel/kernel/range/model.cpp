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

[[nodiscard]] constexpr bool AlgebraContract() {
  constexpr std::array domains{ComputeDomain::I32, ComputeDomain::U32,
                               ComputeDomain::I64, ComputeDomain::U64,
                               ComputeDomain::Fixed};
  for (const ComputeDomain domain : domains) {
    const RangeTraits sum = Traits(RangeOp::Sum, domain);
    const RangeTraits minimum = Traits(RangeOp::Minimum, domain);
    const RangeTraits maximum = Traits(RangeOp::Maximum, domain);
    const std::optional<RangeTraits> saturating =
        RangeTraits::sum_saturating(domain);
    if (!sum.associative() || !sum.has_identity() || !sum.commutative() ||
        !sum.invertible() || sum.idempotent() || sum.ordered() ||
        !minimum.associative() || !minimum.has_identity() ||
        !minimum.commutative() || minimum.invertible() ||
        !minimum.idempotent() || !minimum.ordered() || maximum.invertible() ||
        !maximum.idempotent() || !maximum.ordered()) {
      return false;
    }
    const bool signed_domain = domain == ComputeDomain::I32 ||
                               domain == ComputeDomain::I64 ||
                               domain == ComputeDomain::Fixed;
    const bool unsigned_domain =
        domain == ComputeDomain::U32 || domain == ComputeDomain::U64;
    if (saturating.has_value() != signed_domain ||
        (saturating.has_value() &&
         (saturating->associative() || saturating->invertible() ||
          !saturating->has_identity()))) {
      return false;
    }
    if (sum.signed_domain() != signed_domain ||
        sum.unsigned_domain() != unsigned_domain ||
        sum.fixed_domain() != (domain == ComputeDomain::Fixed)) {
      return false;
    }
  }

  return true;
}

[[nodiscard]] constexpr bool AffineShapeContract() {
  const RangeShape centered = Shape(RangeOp::Sum, 17u, 7u);
  const RangeShape projected =
      AffineShape(RangeOp::Sum, RangeBoundary::Clamp, 17u, 17u, 15u, 1u, 7u);
  const RangeShape strided =
      AffineShape(RangeOp::Minimum, RangeBoundary::Clip, 19u, 7u, 9u, 3u, 2u);
  const RangeShape padded_anchor =
      AffineShape(RangeOp::Maximum, RangeBoundary::Clamp, 5u, 2u, 10u, 8u, 5u);
  const RangeShape wide_window =
      AffineShape(RangeOp::Minimum, RangeBoundary::Clip, 2u, 1u, 6u, 1u, 1u);
  const auto no_intersection = RangeShape::affine(
      Traits(RangeOp::Sum), RangeBoundary::Clip, 5u, 2u, 2u, 10u, 0u, 4u);
  const auto invalid_padding = RangeShape::affine(
      Traits(RangeOp::Sum), RangeBoundary::Clamp, 5u, 1u, 3u, 1u, 3u, 4u);
  const auto overflowing_anchor =
      RangeShape::affine(Traits(RangeOp::Sum), RangeBoundary::Clamp, 2u, 3u, 2u,
                         std::numeric_limits<u64>::max(), 1u, 4u);
  constexpr u64 span_input = std::numeric_limits<u64>::max() / 4u;
  constexpr u64 span_window = 2u * span_input + 1u;
  constexpr u64 span_padding = span_window - 1u;
  const auto span_overflow = RangeShape::affine(
      Traits(RangeOp::Minimum), RangeBoundary::Clamp, span_input, 2u,
      span_window, span_input + span_padding - 1u, span_padding, 4u);
  return centered.input_count() == projected.input_count() &&
         centered.output_count() == projected.output_count() &&
         centered.window_size() == projected.window_size() &&
         centered.stride() == projected.stride() &&
         centered.padding() == projected.padding() &&
         centered.centered_clamp() && projected.centered_clamp() &&
         !strided.centered_clamp() && strided.input_count() == 19u &&
         strided.output_count() == 7u && strided.window_size() == 9u &&
         strided.stride() == 3u && strided.padding() == 2u &&
         strided.right_extent() == 6u && *strided.affine_span() == 27u &&
         padded_anchor.valid() && wide_window.valid() &&
         wide_window.window_size() > 2u * wide_window.input_count() + 1u &&
         !no_intersection.has_value() && !invalid_padding.has_value() &&
         !overflowing_anchor.has_value() && span_overflow.has_value() &&
         !span_overflow->affine_span().has_value();
}

[[nodiscard]] constexpr bool FailClosedContract() {
  const auto invalid_operation = RangeTraits::make(
      static_cast<RangeOp>(255u), ComputeDomain::U32, RangeLaw::ModuloWidth);
  const auto invalid_domain =
      RangeTraits::sum_modulo(static_cast<ComputeDomain>(255u));
  const auto invalid_law = RangeTraits::make(
      RangeOp::Minimum, ComputeDomain::U32, RangeLaw::ModuloWidth);
  const auto invalid_shape = RangeShape::affine(
      Traits(RangeOp::Sum), RangeBoundary::Clamp, 1u, 1u, 1u, 1u, 1u, 4u);
  const auto invalid_domain_width =
      RangeShape::affine(Traits(RangeOp::Sum, ComputeDomain::U64),
                         RangeBoundary::Clamp, 1u, 1u, 3u, 1u, 1u, 4u);
  const auto payload_overflow = RangeShape::affine(
      Traits(RangeOp::Sum), RangeBoundary::Clamp,
      std::numeric_limits<u64>::max() / 4u + 1u, 1u, 1u, 1u, 0u, 4u);
  const auto invalid_caps =
      RangeCaps::gpu(RangeSource::Metal, 0x80u, 256u, 4u, 32768u, 1u,
                     std::numeric_limits<u64>::max(), kAllCandidates);
  const RangePlan unavailable =
      ContractPlanRange(Shape(RangeOp::Sum, 1u, 1u), RangeCaps::unavailable());
  constexpr u64 maximum_u32_count = std::numeric_limits<u64>::max() / 4u;
  constexpr std::uint8_t direct_block =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::BlockPrefixSuffix);
  const RangePlan overflowing_block = ContractPlanRange(
      Shape(RangeOp::Minimum, maximum_u32_count, maximum_u32_count),
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u,
          std::numeric_limits<u64>::max(), direct_block));
  constexpr u64 vulkan_input = std::numeric_limits<u32>::max() / 2u;
  constexpr u64 vulkan_window = 2u * vulkan_input + 1u;
  constexpr u64 vulkan_padding = vulkan_window - 1u;
  const RangeShape oversized_vulkan_span = AffineShape(
      RangeOp::Minimum, RangeBoundary::Clip, vulkan_input, 2u, vulkan_window,
      vulkan_input + vulkan_padding - 1u, vulkan_padding);
  const RangePlan vulkan_span_fallback =
      ContractPlanRange(oversized_vulkan_span,
                        Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
                            std::numeric_limits<u32>::max(), direct_block));
  rund::kernel::u128 ignored = 0u;
  return !invalid_operation.has_value() && !invalid_domain.has_value() &&
         !invalid_law.has_value() && !invalid_shape.has_value() &&
         !invalid_domain_width.has_value() && !invalid_caps.has_value() &&
         !payload_overflow.has_value() && !unavailable.ok() &&
         std::string_view{unavailable.reason()} ==
             "compute_range_aggregate_unavailable" &&
         overflowing_block.ok() &&
         overflowing_block.candidate().disposition() == RangePath::Direct &&
         overflowing_block.legal_candidate_count() == 1u &&
         vulkan_span_fallback.ok() &&
         vulkan_span_fallback.candidate().disposition() == RangePath::Direct &&
         vulkan_span_fallback.legal_candidate_count() == 1u &&
         !range_plan_detail::Multiply(range_plan_detail::kU128Maximum, 2u,
                                      ignored);
}

static_assert(AlgebraContract());
static_assert(AffineShapeContract());
static_assert(FailClosedContract());

} // namespace

bool ModelContract() {
  return AlgebraContract() && AffineShapeContract() && FailClosedContract();
}

} // namespace node_accel_contract::range
