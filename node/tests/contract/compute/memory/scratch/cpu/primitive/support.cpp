#include "support.hpp"

namespace rund_node_memory_contract::cpu_primitive {

rund::compute::detail::CpuRuntimePrimitive
RangeScratchPrimitive(const rund::kernel::WindowPlan &semantic,
                      const rund::node::accel::detail::RangePlan &range) {
  rund::compute::detail::CpuRuntimePrimitive primitive{};
  primitive.kind = rund::compute::detail::Primitive::Window;
  primitive.plan = semantic;
  primitive.range = range;
  return primitive;
}

bool CheckNoCpuPrimitiveScratch(
    const rund::compute::detail::CpuRuntimePrimitive &primitive) {
  using namespace rund::compute::detail;
  const auto scratch_plan = plan_cpu_scratch(primitive);
  if (!scratch_plan) {
    return false;
  }
  CpuPreparedArenaPlan arena_plan{};
  if (!append_cpu_primitive_arena_plan(arena_plan.execution, *scratch_plan) ||
      !seal_cpu_prepared_arena_plan(arena_plan, 4096u)) {
    return false;
  }
  auto arena = make_cpu_prepared_arena(arena_plan);
  if (!arena) {
    return false;
  }
  node_compute_allocation::Start();
  auto prepared = prepare_cpu_scratch(primitive, *scratch_plan, **arena);
  node_compute_allocation::Stop();
  const std::uint64_t allocations = node_compute_allocation::Count();
  if (!prepared || allocations != 0u ||
      !std::holds_alternative<std::monostate>(prepared.value()) ||
      !(*arena)->claims_complete(arena_plan.execution)) {
    return false;
  }
  return (*arena)->payload_host_bytes() == 0u &&
         (*arena)->payload_tile_bytes() == 0u;
}

} // namespace rund_node_memory_contract::cpu_primitive
