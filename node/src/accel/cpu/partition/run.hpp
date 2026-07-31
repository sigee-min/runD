#pragma once

#include <kernel/program/compute/partition/reference.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] inline rund::kernel::PartitionResult RunCpuPartition(
    const rund::kernel::PartitionPlan& plan, const CpuBufferResult& flags,
    const CpuBufferResult& values, const CpuBufferResult& output,
    rund::kernel::u64& false_count, rund::kernel::u64& true_count) {
  const auto* flag_data = flags.buffer->data.data();
  const auto* value_data = values.buffer->data.data();
  auto* output_data = output.buffer->data.data();
  const bool wide_flags = plan.flag_bytes == sizeof(rund::kernel::u64);
  const bool wide_values = plan.value_bytes == sizeof(rund::kernel::u64);
  if (wide_flags && wide_values) {
    return rund::kernel::ReferenceStablePartitionFlagsU64ValuesU64(
        reinterpret_cast<const rund::kernel::u64*>(flag_data),
        reinterpret_cast<const rund::kernel::u64*>(value_data),
        plan.element_count, reinterpret_cast<rund::kernel::u64*>(output_data),
        &false_count, &true_count);
  }
  if (wide_flags) {
    return rund::kernel::ReferenceStablePartitionFlagsU64ValuesU32(
        reinterpret_cast<const rund::kernel::u64*>(flag_data),
        reinterpret_cast<const rund::kernel::u32*>(value_data),
        plan.element_count, reinterpret_cast<rund::kernel::u32*>(output_data),
        &false_count, &true_count);
  }
  if (wide_values) {
    return rund::kernel::ReferenceStablePartitionU64(
        reinterpret_cast<const rund::kernel::u32*>(flag_data),
        reinterpret_cast<const rund::kernel::u64*>(value_data),
        plan.element_count, reinterpret_cast<rund::kernel::u64*>(output_data),
        &false_count, &true_count);
  }
  return rund::kernel::ReferenceStablePartitionU32(
      reinterpret_cast<const rund::kernel::u32*>(flag_data),
      reinterpret_cast<const rund::kernel::u32*>(value_data),
      plan.element_count, reinterpret_cast<rund::kernel::u32*>(output_data),
      &false_count, &true_count);
}

} // namespace rund::node::accel::detail
