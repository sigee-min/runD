#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../scatter/shape.hpp"
#include "buffer/batch.hpp"
#include "scatter/linear.hpp"
#include <kernel/program/compute/scatter/reduce/reference.hpp>

#include <memory>
#include <limits>
#include <mutex>
#include <span>

namespace rund::node::accel::detail {

rund::AccelCheck ExecuteCpuScatter(const rund::AccelDevice &pick,
                                   const rund::kernel::ScatterDesc &desc,
                                   const rund::kernel::ScatterPlan &plan,
                                   const ScatterBinds &bindings) {
  if (!pick.check.ok || !ScatterShapeOk(desc, plan, bindings)) {
    return rund::AccelCheck{false, "compute_scatter_invalid"};
  }
  CpuBufferResult values{};
  CpuBufferResult indices{};
  CpuBufferResult output{};
  CpuResidentReq reqs[] = {
      CpuResidentReq{.ref = bindings.values,
                     .handle = bindings.values_handle,
                     .usage = rund::kernel::kResidentUsageRead,
                     .out = &values},
      CpuResidentReq{.ref = bindings.indices,
                     .handle = bindings.indices_handle,
                     .usage = rund::kernel::kResidentUsageRead,
                     .out = &indices},
      CpuResidentReq{.ref = bindings.output,
                     .handle = bindings.output_handle,
                     .usage = rund::kernel::kResidentUsageWrite,
                     .out = &output},
  };
  LookupCpuResidentBatch(pick, reqs);
  CpuAdapter *const adapter = CpuAdapterFromPick(pick);
  if (!values.check.ok || !indices.check.ok || !output.check.ok ||
      adapter == nullptr) {
    return rund::AccelCheck{false, "compute_scatter_invalid"};
  }
  const auto *const index_data =
      reinterpret_cast<const rund::kernel::u32 *>(indices.buffer->data.data());
  const bool u32 = plan.element == rund::kernel::ScatterElement::U32;
  rund::kernel::ScatterResult result{};
  {
    std::lock_guard<std::mutex> lock{adapter->mutex};
    if (adapter->scatter_scratch == nullptr) {
      adapter->scatter_scratch = std::make_shared<CpuScatterScratch>();
    }
    CpuScatterScratch &scratch = *adapter->scatter_scratch;
    result =
        u32 ? ExecuteLinearScatter(scratch,
                                   reinterpret_cast<const rund::kernel::u32 *>(
                                       values.buffer->data.data()),
                                   index_data,
                                   reinterpret_cast<rund::kernel::u32 *>(
                                       output.buffer->data.data()),
                                   plan.element_count, plan.output_count,
                                   static_cast<std::size_t>(plan.scratch_slots))
            : ExecuteLinearScatter(
                  scratch,
                  reinterpret_cast<const rund::kernel::u64 *>(
                      values.buffer->data.data()),
                  index_data,
                  reinterpret_cast<rund::kernel::u64 *>(
                      output.buffer->data.data()),
                  plan.element_count, plan.output_count,
                  static_cast<std::size_t>(plan.scratch_slots));
  }
  if (!result.ok) {
    return rund::AccelCheck{false, result.reason};
  }
  RecordCpuDispatches(*adapter, plan.pass_count);
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck ExecuteCpuScatterReduce(
    const rund::AccelDevice &pick,
    const rund::kernel::ScatterReducePlan &plan,
    const ScatterReduceBinds &bindings) {
  CpuBufferResult values{}, indices{}, count{}, output{};
  CpuResidentReq reqs[] = {
      {.ref = bindings.values, .handle = bindings.values_handle,
       .usage = rund::kernel::kResidentUsageRead, .out = &values},
      {.ref = bindings.indices, .handle = bindings.indices_handle,
       .usage = rund::kernel::kResidentUsageRead, .out = &indices},
      {.ref = bindings.output, .handle = bindings.output_handle,
       .usage = rund::kernel::kResidentUsageWrite, .out = &output},
  };
  LookupCpuResidentBatch(pick, reqs);
  if (!plan.ok || !values.check.ok || !indices.check.ok || !output.check.ok ||
      bindings.values == nullptr || bindings.indices == nullptr ||
      bindings.output == nullptr) {
    return {false, "compute_scatter_reduce_buffer_invalid"};
  }
  std::uint64_t logical = plan.element_count;
  if (plan.count_source != rund::kernel::ComputeCountSource::Descriptor) {
    CpuResidentReq count_req[]{
        {.ref = bindings.count,
         .handle = bindings.count_handle,
         .usage = rund::kernel::kResidentUsageRead,
         .out = &count}};
    LookupCpuResidentBatch(pick, count_req);
    if (!count.check.ok || bindings.count == nullptr) {
      return {false, "compute_scatter_reduce_buffer_invalid"};
    }
    const auto *raw = count.buffer->data.data() + bindings.count->offset_bytes;
    logical = plan.count_source == rund::kernel::ComputeCountSource::BufferU64
                  ? *reinterpret_cast<const std::uint64_t *>(raw)
                  : *reinterpret_cast<const std::uint32_t *>(raw);
  }
  const auto *value_raw = values.buffer->data.data() + bindings.values->offset_bytes;
  const auto *index_raw = indices.buffer->data.data() + bindings.indices->offset_bytes;
  auto *output_raw = output.buffer->data.data() + bindings.output->offset_bytes;
  const auto *index = reinterpret_cast<const rund::kernel::u32 *>(index_raw);
  CpuAdapter *const adapter = CpuAdapterFromPick(pick);
  if (adapter == nullptr ||
      plan.element_count > std::numeric_limits<std::size_t>::max()) {
    return {false, "compute_scatter_reduce_buffer_invalid"};
  }
  std::lock_guard<std::mutex> lock{adapter->mutex};
  adapter->scatter_reduce_indices.resize(
      static_cast<std::size_t>(plan.element_count));
  auto *const sorted_indices = adapter->scatter_reduce_indices.data();
  const std::size_t sorted_index_capacity =
      adapter->scatter_reduce_indices.size();
  rund::kernel::ScatterReduceResult result{};
  if (plan.domain == rund::kernel::ComputeDomain::I32) {
    result = rund::kernel::ReferenceScatterReduceI32(
        reinterpret_cast<const rund::kernel::i32 *>(value_raw), index,
        reinterpret_cast<rund::kernel::i32 *>(output_raw), logical, plan,
        sorted_indices, sorted_index_capacity);
  } else if (plan.domain == rund::kernel::ComputeDomain::U32) {
    result = rund::kernel::ReferenceScatterReduceU32(
        reinterpret_cast<const rund::kernel::u32 *>(value_raw), index,
        reinterpret_cast<rund::kernel::u32 *>(output_raw), logical, plan,
        sorted_indices, sorted_index_capacity);
  } else if (plan.domain == rund::kernel::ComputeDomain::I64) {
    result = rund::kernel::ReferenceScatterReduceI64(
        reinterpret_cast<const rund::kernel::i64 *>(value_raw), index,
        reinterpret_cast<rund::kernel::i64 *>(output_raw), logical, plan,
        sorted_indices, sorted_index_capacity);
  } else if (plan.domain == rund::kernel::ComputeDomain::U64) {
    result = rund::kernel::ReferenceScatterReduceU64(
        reinterpret_cast<const rund::kernel::u64 *>(value_raw), index,
        reinterpret_cast<rund::kernel::u64 *>(output_raw), logical, plan,
        sorted_indices, sorted_index_capacity);
  } else if (plan.element_bytes == 4u) {
    result = rund::kernel::ReferenceScatterReduceFixedI32(
        reinterpret_cast<const rund::kernel::i32 *>(value_raw), index,
        reinterpret_cast<rund::kernel::i32 *>(output_raw), logical, plan,
        sorted_indices, sorted_index_capacity);
  } else {
    result = rund::kernel::ReferenceScatterReduceFixedI64(
        reinterpret_cast<const rund::kernel::i64 *>(value_raw), index,
        reinterpret_cast<rund::kernel::i64 *>(output_raw), logical, plan,
        sorted_indices, sorted_index_capacity);
  }
  return {result.ok, result.reason};
}

} // namespace rund::node::accel::detail
