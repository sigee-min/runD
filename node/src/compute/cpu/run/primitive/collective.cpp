#include "../../state/program.hpp"
#include "local.hpp"

#include "../../../job/state.hpp"
#include "../../graph.hpp"

#include <kernel/program/compute/segmented/reduce/reference.hpp>
#include <kernel/program/compute/segmented/scan/reference.hpp>

#include <cstdint>

namespace rund::compute::detail {

Status run_cpu_primitive_collective(PrimitiveContext &context) {
  if (context.primitive.kind == Primitive::SegmentedScan) {
    const auto &plan =
        std::get<kernel::SegmentedScanPlan>(context.primitive.plan);
    const CpuRuntimeGraph &graph = *context.job.program->cpu_graph->runtime;
    const Type value_type =
        graph.values[context.primitive.inputs.front() - 1u].type;
    const auto *heads =
        reinterpret_cast<const kernel::u32 *>(context.port(1u).data);
    kernel::SegmentedScanResult result{};
    if (value_type == Type::I32 || value_type == Type::FixedLane32) {
      result = kernel::ReferenceSignedSegmentedScan(
          reinterpret_cast<const std::int32_t *>(context.port(0u).data), heads,
          reinterpret_cast<std::int32_t *>(context.port(2u).data),
          plan.element_count, plan.op == kernel::SegmentedScanOp::InclusiveSum);
    } else if (value_type == Type::I64 || value_type == Type::FixedLane64) {
      result = kernel::ReferenceSignedSegmentedScan(
          reinterpret_cast<const std::int64_t *>(context.port(0u).data), heads,
          reinterpret_cast<std::int64_t *>(context.port(2u).data),
          plan.element_count, plan.op == kernel::SegmentedScanOp::InclusiveSum);
    } else if (plan.element == kernel::SegmentedScanElement::U32 &&
               plan.op == kernel::SegmentedScanOp::ExclusiveSum) {
      result = kernel::ReferenceExclusiveSegmentedScanU32(
          reinterpret_cast<const kernel::u32 *>(context.port(0u).data), heads,
          reinterpret_cast<kernel::u32 *>(context.port(2u).data),
          plan.element_count);
    } else if (plan.element == kernel::SegmentedScanElement::U32) {
      result = kernel::ReferenceInclusiveSegmentedScanU32(
          reinterpret_cast<const kernel::u32 *>(context.port(0u).data), heads,
          reinterpret_cast<kernel::u32 *>(context.port(2u).data),
          plan.element_count);
    } else if (plan.op == kernel::SegmentedScanOp::ExclusiveSum) {
      result = kernel::ReferenceExclusiveSegmentedScanU64(
          reinterpret_cast<const kernel::u64 *>(context.port(0u).data), heads,
          reinterpret_cast<kernel::u64 *>(context.port(2u).data),
          plan.element_count);
    } else {
      result = kernel::ReferenceInclusiveSegmentedScanU64(
          reinterpret_cast<const kernel::u64 *>(context.port(0u).data), heads,
          reinterpret_cast<kernel::u64 *>(context.port(2u).data),
          plan.element_count);
    }
    return finish_cpu_primitive(result.ok, result.reason);
  }

  if (context.primitive.kind == Primitive::SegmentedReduce) {
    const auto &plan =
        std::get<kernel::SegmentedReducePlan>(context.primitive.plan);
    const auto op = plan.op;
    const CpuRuntimeGraph &graph = *context.job.program->cpu_graph->runtime;
    const Type value_type =
        graph.values[context.primitive.inputs.front() - 1u].type;
    const auto *heads =
        reinterpret_cast<const kernel::u32 *>(context.port(1u).data);
    kernel::SegmentedReduceResult result{};
    if (value_type == Type::I32 || value_type == Type::FixedLane32) {
      result = kernel::ReferenceSignedSegmentedReduce(
          reinterpret_cast<const std::int32_t *>(context.port(0u).data), heads,
          reinterpret_cast<std::int32_t *>(context.port(2u).data),
          plan.element_count, op);
    } else if (value_type == Type::I64 || value_type == Type::FixedLane64) {
      result = kernel::ReferenceSignedSegmentedReduce(
          reinterpret_cast<const std::int64_t *>(context.port(0u).data), heads,
          reinterpret_cast<std::int64_t *>(context.port(2u).data),
          plan.element_count, op);
    } else if (plan.element == kernel::ReduceElement::U32) {
      const auto *values =
          reinterpret_cast<const kernel::u32 *>(context.port(0u).data);
      auto *output = reinterpret_cast<kernel::u32 *>(context.port(2u).data);
      switch (op) {
      case kernel::ReduceOp::Sum:
        result = kernel::ReferenceSegmentedReduceSumU32(values, heads, output,
                                                        plan.element_count);
        break;
      case kernel::ReduceOp::CountNonzero:
        result = kernel::ReferenceSegmentedReduceCountNonzeroU32(
            values, heads, output, plan.element_count);
        break;
      case kernel::ReduceOp::Min:
        result = kernel::ReferenceSegmentedReduceMinU32(values, heads, output,
                                                        plan.element_count);
        break;
      case kernel::ReduceOp::Max:
        result = kernel::ReferenceSegmentedReduceMaxU32(values, heads, output,
                                                        plan.element_count);
        break;
      }
    } else {
      const auto *values =
          reinterpret_cast<const kernel::u64 *>(context.port(0u).data);
      auto *output = reinterpret_cast<kernel::u64 *>(context.port(2u).data);
      switch (op) {
      case kernel::ReduceOp::Sum:
        result = kernel::ReferenceSegmentedReduceSumU64(values, heads, output,
                                                        plan.element_count);
        break;
      case kernel::ReduceOp::CountNonzero:
        result = kernel::ReferenceSegmentedReduceCountNonzeroU64(
            values, heads, output, plan.element_count);
        break;
      case kernel::ReduceOp::Min:
        result = kernel::ReferenceSegmentedReduceMinU64(values, heads, output,
                                                        plan.element_count);
        break;
      case kernel::ReduceOp::Max:
        result = kernel::ReferenceSegmentedReduceMaxU64(values, heads, output,
                                                        plan.element_count);
        break;
      }
    }
    return finish_cpu_primitive(result.ok, result.reason);
  }

  return finish_cpu_primitive(false, "compute_primitive_unsupported");
}

} // namespace rund::compute::detail
