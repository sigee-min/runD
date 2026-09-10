#include "support.hpp"

#include <optional>

namespace rund_node_memory_contract::cpu_primitive {

int CheckRange() {
  using namespace rund::compute::detail;
  using namespace rund::kernel;
  using namespace rund::node::accel::detail;
  constexpr u64 count = 257u;
  constexpr u64 window = 129u;
  constexpr u64 output_count = count - window + 1u;
  const auto sum_traits = RangeTraits::sum_modulo(ComputeDomain::U32);
  const auto min_traits = RangeTraits::minimum(ComputeDomain::U32);
  if (!sum_traits || !min_traits) {
    return 1;
  }
  const auto sum_shape =
      RangeShape::affine(*sum_traits, RangeBoundary::Clip, count, output_count,
                         window, 1u, 0u, sizeof(u32));
  const auto min_shape =
      RangeShape::affine(*min_traits, RangeBoundary::Clip, count, output_count,
                         window, 1u, 0u, sizeof(u32));
  if (!sum_shape || !min_shape) {
    return 2;
  }
  const RangePlan prefix = PlanRange(*sum_shape, RangeCaps::cpu());
  const RangePlan bidirectional = PlanRange(*min_shape, RangeCaps::cpu());
  if (!prefix.ok() || !bidirectional.ok() ||
      prefix.candidate().disposition() != RangePath::PrefixDifference ||
      bidirectional.candidate().disposition() != RangePath::BlockPrefixSuffix ||
      prefix.temporary_count() != 1u || bidirectional.temporary_count() != 2u ||
      prefix.temporary(0u).bytes != count * sizeof(u32) ||
      bidirectional.temporary(0u).bytes != count * sizeof(u32) ||
      bidirectional.temporary(1u).bytes != count * sizeof(u32)) {
    return 3;
  }

  const WindowPlan sum = PlanWindow(WindowDesc{
      .op = WindowOp::Sum,
      .element = WindowElement::U32,
      .boundary = WindowBoundary::Clip,
      .domain = ComputeDomain::U32,
      .input_count = count,
      .output_count = output_count,
      .window_size = window,
      .stride = 1u,
      .pad_left = 0u,
  });
  const WindowPlan minimum = PlanWindow(WindowDesc{
      .op = WindowOp::Min,
      .element = WindowElement::U32,
      .boundary = WindowBoundary::Clip,
      .domain = ComputeDomain::U32,
      .input_count = count,
      .output_count = output_count,
      .window_size = window,
      .stride = 1u,
      .pad_left = 0u,
  });
  const CpuRuntimePrimitive prefix_primitive =
      RangeScratchPrimitive(sum, prefix);
  const CpuRuntimePrimitive bidirectional_primitive =
      RangeScratchPrimitive(minimum, bidirectional);
  const auto prefix_scratch = plan_cpu_scratch(prefix_primitive);
  const auto bidirectional_scratch = plan_cpu_scratch(bidirectional_primitive);
  if (!sum.ok || !minimum.ok || !prefix_scratch || !bidirectional_scratch ||
      prefix_scratch->shape != CpuPrimitiveScratchShape::RangeU32 ||
      bidirectional_scratch->shape != CpuPrimitiveScratchShape::RangeU32 ||
      prefix_scratch->counts[0] != count || prefix_scratch->counts[1] != 0u ||
      bidirectional_scratch->counts[0] != count ||
      bidirectional_scratch->counts[1] != count) {
    return 4;
  }

  CpuPreparedArenaPlan arena_plan{};
  if (!append_cpu_primitive_arena_plan(arena_plan.execution, *prefix_scratch) ||
      !append_cpu_primitive_arena_plan(arena_plan.execution,
                                       *bidirectional_scratch) ||
      arena_plan.execution.primitive_u32_count != 2u * count ||
      arena_plan.execution.primitive_object_payload_bytes !=
          2u * sizeof(CpuRangeScratch<u32>) ||
      !seal_cpu_prepared_arena_plan(arena_plan, 4096u)) {
    return 5;
  }
  auto arena = make_cpu_prepared_arena(arena_plan);
  if (!arena || (*arena)->payload_tile_bytes() != 2u * count * sizeof(u32) ||
      (*arena)->payload_host_bytes() != 2u * sizeof(CpuRangeScratch<u32>)) {
    return 6;
  }

  node_compute_allocation::Start();
  auto prefix_prepared =
      prepare_cpu_scratch(prefix_primitive, *prefix_scratch, **arena);
  auto bidirectional_prepared = prepare_cpu_scratch(
      bidirectional_primitive, *bidirectional_scratch, **arena);
  node_compute_allocation::Stop();
  auto *const prefix_owner =
      prefix_prepared
          ? cpu_primitive_scratch<CpuRangeScratch<u32>>(*prefix_prepared)
          : nullptr;
  auto *const bidirectional_owner =
      bidirectional_prepared
          ? cpu_primitive_scratch<CpuRangeScratch<u32>>(*bidirectional_prepared)
          : nullptr;
  if (node_compute_allocation::Count() != 0u || prefix_owner == nullptr ||
      bidirectional_owner == nullptr || prefix_owner->first.size() != count ||
      !prefix_owner->second.empty() ||
      bidirectional_owner->first.size() != count ||
      bidirectional_owner->second.size() != count ||
      prefix_owner->first.data() != bidirectional_owner->first.data() ||
      !(*arena)->claims_complete(arena_plan.execution)) {
    return 7;
  }

  ComputeFixedFormat wrap_format = ScratchFixedFormat(sizeof(i32));
  wrap_format.overflow = ComputeOverflow::Wrap;
  const auto fixed_traits = RangeTraits::sum_modulo(ComputeDomain::Fixed);
  const auto fixed_shape =
      fixed_traits
          ? RangeShape::affine(*fixed_traits, RangeBoundary::Clip, count,
                               output_count, window, 1u, 0u, sizeof(i32))
          : std::nullopt;
  const RangePlan fixed_prefix =
      fixed_shape ? PlanRange(*fixed_shape, RangeCaps::cpu())
                  : RangePlan::rejected("compute_range_shape_invalid");
  const WindowPlan fixed_sum = PlanWindow(WindowDesc{
      .op = WindowOp::Sum,
      .element = WindowElement::U32,
      .boundary = WindowBoundary::Clip,
      .domain = ComputeDomain::Fixed,
      .fixed_format = wrap_format,
      .input_count = count,
      .output_count = output_count,
      .window_size = window,
      .stride = 1u,
      .pad_left = 0u,
  });
  if (!fixed_prefix.ok() || !fixed_sum.ok ||
      fixed_prefix.candidate().disposition() != RangePath::PrefixDifference ||
      !CheckArenaCpuPrimitiveScratch<CpuRangeScratch<i32>>(
          RangeScratchPrimitive(fixed_sum, fixed_prefix),
          count * sizeof(i32))) {
    return 8;
  }
  return 0;
}

} // namespace rund_node_memory_contract::cpu_primitive
