#include "local.hpp"

#include "../../graph.hpp"

#include <kernel/program/compute/compact/reference.hpp>
#include <kernel/program/compute/histogram/reference.hpp>
#include <kernel/program/compute/partition/reference.hpp>

namespace rund::compute::detail {

Status run_cpu_primitive_reference(PrimitiveContext &context) {
  switch (context.primitive.kind) {
  case Primitive::Compact: {
    const auto &plan = std::get<kernel::CompactPlan>(context.primitive.plan);
    kernel::u64 output_count = 0u;
    const auto result = kernel::ReferenceCompactIdsU32(
        reinterpret_cast<const kernel::u32 *>(context.port(0u).data),
        plan.element_count, plan.output_capacity,
        reinterpret_cast<kernel::u32 *>(context.port(1u).data), &output_count);
    return finish_cpu_primitive(result.ok, result.reason);
  }
  case Primitive::Histogram: {
    const auto &plan = std::get<kernel::HistogramPlan>(context.primitive.plan);
    const auto result = kernel::ReferenceHistogramU32(
        reinterpret_cast<const kernel::u32 *>(context.port(0u).data),
        reinterpret_cast<kernel::u32 *>(context.port(1u).data),
        plan.element_count, plan.bin_count);
    return finish_cpu_primitive(result.ok, result.reason);
  }
  case Primitive::Partition: {
    const auto &plan = std::get<kernel::PartitionPlan>(context.primitive.plan);
    kernel::u64 false_count = 0u;
    kernel::u64 true_count = 0u;
    const bool wide_flags = plan.flag_bytes == sizeof(kernel::u64);
    const bool wide_values = plan.value_bytes == sizeof(kernel::u64);
    const auto result =
        wide_flags && wide_values
            ? kernel::ReferenceStablePartitionFlagsU64ValuesU64(
                  reinterpret_cast<const kernel::u64 *>(context.port(0u).data),
                  reinterpret_cast<const kernel::u64 *>(context.port(1u).data),
                  plan.element_count,
                  reinterpret_cast<kernel::u64 *>(context.port(2u).data),
                  &false_count, &true_count)
        : wide_flags
            ? kernel::ReferenceStablePartitionFlagsU64ValuesU32(
                  reinterpret_cast<const kernel::u64 *>(context.port(0u).data),
                  reinterpret_cast<const kernel::u32 *>(context.port(1u).data),
                  plan.element_count,
                  reinterpret_cast<kernel::u32 *>(context.port(2u).data),
                  &false_count, &true_count)
        : wide_values
            ? kernel::ReferenceStablePartitionU64(
                  reinterpret_cast<const kernel::u32 *>(context.port(0u).data),
                  reinterpret_cast<const kernel::u64 *>(context.port(1u).data),
                  plan.element_count,
                  reinterpret_cast<kernel::u64 *>(context.port(2u).data),
                  &false_count, &true_count)
            : kernel::ReferenceStablePartitionU32(
                  reinterpret_cast<const kernel::u32 *>(context.port(0u).data),
                  reinterpret_cast<const kernel::u32 *>(context.port(1u).data),
                  plan.element_count,
                  reinterpret_cast<kernel::u32 *>(context.port(2u).data),
                  &false_count, &true_count);
    return finish_cpu_primitive(result.ok, result.reason);
  }
  default:
    return finish_cpu_primitive(false, "compute_primitive_unsupported");
  }
}

} // namespace rund::compute::detail
