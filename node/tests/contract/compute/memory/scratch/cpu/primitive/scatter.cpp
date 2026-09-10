#include "support.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace rund_node_memory_contract::cpu_primitive {
namespace {

[[nodiscard]] bool CheckSharedScatterEpoch() {
  using namespace rund::compute::detail;
  using namespace rund::kernel;
  const ScatterPlan small_plan = PlanScatter(ScatterDesc{
      .element = ScatterElement::U32,
      .element_count = 3u,
      .output_count = 4u,
  });
  const ScatterPlan large_plan = PlanScatter(ScatterDesc{
      .element = ScatterElement::U32,
      .element_count = 9u,
      .output_count = 16u,
  });
  const CpuRuntimePrimitive small =
      ScratchPrimitive(Primitive::Scatter, small_plan);
  const CpuRuntimePrimitive large =
      ScratchPrimitive(Primitive::Scatter, large_plan);
  const auto small_scratch_plan = plan_cpu_scratch(small);
  const auto large_scratch_plan = plan_cpu_scratch(large);
  if (!small_plan.ok || !large_plan.ok || !small_scratch_plan ||
      !large_scratch_plan ||
      small_scratch_plan->counts[0] >= large_scratch_plan->counts[0]) {
    return false;
  }

  CpuPreparedArenaPlan arena_plan{};
  if (!append_cpu_primitive_arena_plan(arena_plan.execution,
                                       *small_scratch_plan) ||
      !append_cpu_primitive_arena_plan(arena_plan.execution,
                                       *large_scratch_plan) ||
      !seal_cpu_prepared_arena_plan(arena_plan, 4096u)) {
    return false;
  }
  CpuPreparedArenaPlan tampered_segment = arena_plan;
  ++tampered_segment.scatter_marks.offset_bytes;
  const auto rejected_segment = make_cpu_prepared_arena(tampered_segment);
  CpuPreparedArenaPlan tampered_execution = arena_plan;
  --tampered_execution.execution.scatter_slot_count;
  const auto rejected_execution = make_cpu_prepared_arena(tampered_execution);
  if (rejected_segment ||
      rejected_segment.reason() != rund::compute::Reason::CpuRuntimeInvalid ||
      rejected_execution ||
      rejected_execution.reason() != rund::compute::Reason::CpuRuntimeInvalid) {
    return false;
  }
  auto arena = make_cpu_prepared_arena(arena_plan);
  if (!arena) {
    return false;
  }
  auto first = prepare_cpu_scratch(small, *small_scratch_plan, **arena);
  auto second = prepare_cpu_scratch(large, *large_scratch_plan, **arena);
  auto *const small_scratch =
      first ? cpu_primitive_scratch<CpuScatterPrimitiveScratch>(*first)
            : nullptr;
  auto *const large_scratch =
      second ? cpu_primitive_scratch<CpuScatterPrimitiveScratch>(*second)
             : nullptr;
  if (small_scratch == nullptr || large_scratch == nullptr ||
      small_scratch->keys.data() != large_scratch->keys.data() ||
      small_scratch->marks.data() != large_scratch->marks.data() ||
      small_scratch->epoch != large_scratch->epoch ||
      small_scratch->mark_capacity.data() !=
          large_scratch->mark_capacity.data() ||
      small_scratch->mark_capacity.size() != large_scratch->marks.size() ||
      !(*arena)->claims_complete(arena_plan.execution)) {
    return false;
  }

  std::fill(small_scratch->mark_capacity.begin(),
            small_scratch->mark_capacity.end(),
            std::numeric_limits<u32>::max());
  *small_scratch->epoch = std::numeric_limits<u32>::max();
  constexpr std::array<u32, 1u> values{7u};
  constexpr std::array<u32, 1u> indices{1u};
  std::array<u32, 16u> small_output{};
  std::array<u32, 16u> large_output{};
  const ScatterResult first_result =
      rund::node::accel::detail::ExecuteLinearScatter(
          *small_scratch, values.data(), indices.data(), small_output.data(),
          values.size(), 4u, small_scratch->marks.size());
  const bool tail_cleared = std::all_of(
      small_scratch->mark_capacity.begin() + small_scratch->marks.size(),
      small_scratch->mark_capacity.end(),
      [](const u32 mark) { return mark == 0u; });
  const ScatterResult second_result =
      rund::node::accel::detail::ExecuteLinearScatter(
          *large_scratch, values.data(), indices.data(), large_output.data(),
          values.size(), 16u, large_scratch->marks.size());
  return first_result.ok && second_result.ok && tail_cleared &&
         small_output[1u] == values[0u] && large_output[1u] == values[0u] &&
         *small_scratch->epoch == 2u;
}

} // namespace

int CheckScatter() {
  using namespace rund::compute::detail;
  using namespace rund::kernel;
  const SortPlan sort32 = PlanSort(SortDesc{
      .key = SortKey::U32,
      .value = SortValue::IdentityU32,
      .element_count = 5u,
  });
  const SortPlan sort64 = PlanSort(SortDesc{
      .key = SortKey::U64,
      .value = SortValue::IdentityU32,
      .element_count = 5u,
  });
  if (!sort32.ok ||
      !CheckArenaCpuPrimitiveScratch<
          CpuSortPrimitiveScratch<rund::kernel::u32>>(
          ScratchPrimitive(Primitive::Sort, sort32),
          5u * (sizeof(rund::kernel::u32) + sizeof(rund::kernel::u32)))) {
    return 3;
  }
  if (!sort64.ok ||
      !CheckArenaCpuPrimitiveScratch<
          CpuSortPrimitiveScratch<rund::kernel::u64>>(
          ScratchPrimitive(Primitive::Argsort, sort64),
          5u * (sizeof(rund::kernel::u64) + sizeof(rund::kernel::u32)))) {
    return 4;
  }

  const ScatterPlan scatter = PlanScatter(ScatterDesc{
      .element = ScatterElement::U32,
      .element_count = 5u,
      .output_count = 8u,
  });
  if (!scatter.ok || scatter.scratch_slots != 16u ||
      !CheckArenaCpuPrimitiveScratch<CpuScatterPrimitiveScratch>(
          ScratchPrimitive(Primitive::Scatter, scatter),
          16u * (sizeof(rund::kernel::u32) + sizeof(rund::kernel::u32)))) {
    return 5;
  }
  if (!CheckSharedScatterEpoch()) {
    return 6;
  }
  return 0;
}

} // namespace rund_node_memory_contract::cpu_primitive
