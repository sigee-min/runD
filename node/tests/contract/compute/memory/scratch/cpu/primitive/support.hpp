#pragma once

#include "local.hpp"

#include "../../../../allocation.hpp"

#include "../../../../../../../src/accel/cpu/scatter/linear.hpp"
#include "../../../../../../../src/accel/range_aggregate/plan.hpp"
#include "../../../../../../../src/compute/cpu/graph.hpp"
#include "../../../../../../../src/compute/cpu/prepared.hpp"
#include "../../../../../../../src/compute/cpu/scratch.hpp"
#include "../../../../../../../src/compute/memory/cpu.hpp"

#include <kernel/program/compute/compact/plan.hpp>
#include <kernel/program/compute/factor/plan.hpp>
#include <kernel/program/compute/gather/plan.hpp>
#include <kernel/program/compute/histogram/plan.hpp>
#include <kernel/program/compute/matrix/plan.hpp>
#include <kernel/program/compute/partition/plan.hpp>
#include <kernel/program/compute/reduce/plan.hpp>
#include <kernel/program/compute/scatter/plan.hpp>
#include <kernel/program/compute/segmented/reduce/plan.hpp>
#include <kernel/program/compute/segmented/scan/plan.hpp>
#include <kernel/program/compute/solve/plan.hpp>
#include <kernel/program/compute/sort/plan.hpp>
#include <kernel/program/compute/spectrum/plan.hpp>
#include <kernel/program/compute/stencil/plan.hpp>
#include <kernel/program/compute/transform/plan.hpp>
#include <kernel/program/compute/window/plan.hpp>

#include <cstdint>

namespace rund_node_memory_contract::cpu_primitive {

[[nodiscard]] constexpr rund::kernel::ComputeFixedFormat
ScratchFixedFormat(const std::uint32_t element_bytes) noexcept {
  return rund::kernel::ComputeFixedFormat{
      .integer_bits = 1u,
      .fraction_bits = static_cast<rund::kernel::u8>(element_bytes * 8u - 1u),
      .rounding = rund::kernel::ComputeRounding::NearestEven,
      .overflow = rund::kernel::ComputeOverflow::Saturate,
      .approximation = rund::kernel::ComputeApproximation::Deterministic,
  };
}

template <class Plan>
[[nodiscard]] rund::compute::detail::CpuRuntimePrimitive
ScratchPrimitive(const rund::compute::detail::Primitive kind,
                 const Plan &plan) {
  rund::compute::detail::CpuRuntimePrimitive primitive{};
  primitive.kind = kind;
  primitive.plan = plan;
  return primitive;
}

[[nodiscard]] rund::compute::detail::CpuRuntimePrimitive
RangeScratchPrimitive(const rund::kernel::WindowPlan &semantic,
                      const rund::node::accel::detail::RangePlan &range);

[[nodiscard]] bool CheckNoCpuPrimitiveScratch(
    const rund::compute::detail::CpuRuntimePrimitive &primitive);

template <class Scratch>
[[nodiscard]] bool CheckArenaCpuPrimitiveScratch(
    const rund::compute::detail::CpuRuntimePrimitive &primitive,
    const std::uint64_t expected_tile_bytes) {
  using namespace rund::compute::detail;
  const auto scratch_plan = plan_cpu_scratch(primitive);
  if (!scratch_plan || scratch_plan->host_bytes != sizeof(Scratch)) {
    return false;
  }
  CpuPreparedArenaPlan arena_plan{};
  if (!append_cpu_primitive_arena_plan(arena_plan.execution, *scratch_plan) ||
      !seal_cpu_prepared_arena_plan(arena_plan, 4096u)) {
    return false;
  }
  auto arena = make_cpu_prepared_arena(arena_plan);
  const std::uint64_t shared_tile_bytes =
      scratch_plan->shape == CpuPrimitiveScratchShape::Scatter
          ? sizeof(rund::kernel::u32)
          : 0u;
  if (!arena || (*arena)->payload_host_bytes() != sizeof(Scratch) ||
      (*arena)->payload_tile_bytes() !=
          expected_tile_bytes + shared_tile_bytes) {
    return false;
  }
  node_compute_allocation::Start();
  auto prepared = prepare_cpu_scratch(primitive, *scratch_plan, **arena);
  node_compute_allocation::Stop();
  const std::uint64_t allocations = node_compute_allocation::Count();
  if (!prepared || allocations != 0u ||
      cpu_primitive_scratch<Scratch>(prepared.value()) == nullptr ||
      !(*arena)->claims_complete(arena_plan.execution)) {
    return false;
  }
  return true;
}

} // namespace rund_node_memory_contract::cpu_primitive
