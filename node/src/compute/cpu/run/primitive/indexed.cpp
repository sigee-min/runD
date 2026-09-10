#include "local.hpp"

#include "../../graph.hpp"
#include "../../scratch.hpp"

#include <kernel/program/compute/gather/reference.hpp>
#include <kernel/program/compute/scatter/reduce/reference.hpp>
#include <rund/counter.hpp>

#include <algorithm>
#include <cstdint>

namespace rund::compute::detail {

Status run_cpu_primitive_indexed(PrimitiveContext &context) {
  if (context.primitive.kind == Primitive::Gather) {
    const auto &plan = std::get<kernel::GatherPlan>(context.primitive.plan);
    kernel::u32 logical_count = static_cast<kernel::u32>(plan.element_count);
    const bool bounded = context.primitive.inputs.size() == 3u;
    if (bounded) {
      std::uint32_t count = 0u;
      const Status count_status = read_cpu_primitive_count(
          context, context.primitive.inputs[2u], plan.element_count, count);
      if (!count_status) {
        return count_status;
      }
      logical_count = static_cast<kernel::u32>(count);
    }
    const std::size_t output = bounded ? 3u : 2u;
    const auto result =
        plan.element == kernel::GatherElement::U32
            ? kernel::ReferenceGatherU32(
                  reinterpret_cast<const kernel::u32 *>(context.port(0u).data),
                  reinterpret_cast<const kernel::u32 *>(context.port(1u).data),
                  reinterpret_cast<kernel::u32 *>(context.port(output).data),
                  logical_count, plan.source_count)
            : kernel::ReferenceGatherU64(
                  reinterpret_cast<const kernel::u64 *>(context.port(0u).data),
                  reinterpret_cast<const kernel::u32 *>(context.port(1u).data),
                  reinterpret_cast<kernel::u64 *>(context.port(output).data),
                  logical_count, plan.source_count);
    if (!result.ok) {
      context.run.overflow_ordinal =
          std::min(context.run.overflow_ordinal, result.first_invalid_index);
    }
    return finish_cpu_primitive(result.ok, result.reason);
  }

  if (context.primitive.kind == Primitive::ScatterReduce) {
    const auto &plan =
        std::get<kernel::ScatterReducePlan>(context.primitive.plan);
    auto *const scratch =
        cpu_primitive_scratch<CpuScatterReducePrimitiveScratch>(
            cpu_step_scratch(context.run, context.step));
    if (scratch == nullptr) {
      return finish_cpu_primitive(false,
                                  "compute_scatter_reduce_buffer_invalid");
    }
    kernel::u32 logical_count = static_cast<kernel::u32>(plan.element_count);
    const bool bounded = context.primitive.inputs.size() == 3u;
    if (bounded) {
      std::uint32_t count = 0u;
      const Status count_status = read_cpu_primitive_count(
          context, context.primitive.inputs[2u], plan.element_count, count);
      if (!count_status) {
        context.run.overflow_ordinal =
            std::min(context.run.overflow_ordinal,
                     static_cast<std::uint64_t>(plan.element_count));
        return count_status;
      }
      logical_count = static_cast<kernel::u32>(count);
    }
    const RawCpuBuffer &values = context.port(0u);
    const RawCpuBuffer &indices = context.port(1u);
    const RawCpuBuffer &output = context.port(bounded ? 3u : 2u);
    kernel::ScatterReduceResult result{};
    switch (plan.domain) {
    case kernel::ComputeDomain::I32:
      result = kernel::ReferenceScatterReduceI32(
          reinterpret_cast<const kernel::i32 *>(values.data),
          reinterpret_cast<const kernel::u32 *>(indices.data),
          reinterpret_cast<kernel::i32 *>(output.data), logical_count, plan,
          scratch->sorted_indices.data(), scratch->sorted_indices.size());
      break;
    case kernel::ComputeDomain::U32:
      result = kernel::ReferenceScatterReduceU32(
          reinterpret_cast<const kernel::u32 *>(values.data),
          reinterpret_cast<const kernel::u32 *>(indices.data),
          reinterpret_cast<kernel::u32 *>(output.data), logical_count, plan,
          scratch->sorted_indices.data(), scratch->sorted_indices.size());
      break;
    case kernel::ComputeDomain::I64:
      result = kernel::ReferenceScatterReduceI64(
          reinterpret_cast<const kernel::i64 *>(values.data),
          reinterpret_cast<const kernel::u32 *>(indices.data),
          reinterpret_cast<kernel::i64 *>(output.data), logical_count, plan,
          scratch->sorted_indices.data(), scratch->sorted_indices.size());
      break;
    case kernel::ComputeDomain::U64:
      result = kernel::ReferenceScatterReduceU64(
          reinterpret_cast<const kernel::u64 *>(values.data),
          reinterpret_cast<const kernel::u32 *>(indices.data),
          reinterpret_cast<kernel::u64 *>(output.data), logical_count, plan,
          scratch->sorted_indices.data(), scratch->sorted_indices.size());
      break;
    case kernel::ComputeDomain::Fixed:
      result = plan.element_bytes == sizeof(kernel::i64)
                   ? kernel::ReferenceScatterReduceFixedI64(
                         reinterpret_cast<const kernel::i64 *>(values.data),
                         reinterpret_cast<const kernel::u32 *>(indices.data),
                         reinterpret_cast<kernel::i64 *>(output.data),
                         logical_count, plan, scratch->sorted_indices.data(),
                         scratch->sorted_indices.size())
                   : kernel::ReferenceScatterReduceFixedI32(
                         reinterpret_cast<const kernel::i32 *>(values.data),
                         reinterpret_cast<const kernel::u32 *>(indices.data),
                         reinterpret_cast<kernel::i32 *>(output.data),
                         logical_count, plan, scratch->sorted_indices.data(),
                         scratch->sorted_indices.size());
      break;
    }
    if (result.ok) {
      context.run.conflict_count = ::rund::detail::counter::SaturatingAdd(
          context.run.conflict_count, result.conflict_count);
    } else {
      context.run.overflow_ordinal =
          std::min(context.run.overflow_ordinal, result.first_rejected_ordinal);
    }
    return finish_cpu_primitive(result.ok, result.reason);
  }

  return finish_cpu_primitive(false, "compute_primitive_unsupported");
}

} // namespace rund::compute::detail
