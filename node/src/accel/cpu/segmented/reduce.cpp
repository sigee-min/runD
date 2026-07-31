#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../domain.hpp"
#include "../../segmented/reduce/shape.hpp"
#include "../buffer/batch.hpp"

#include <kernel/program/compute/segmented/reduce/reference.hpp>

namespace rund::node::accel::detail {
namespace {

using rund::kernel::ReduceOp;
using rund::kernel::SegmentedReduceResult;

[[nodiscard]] SegmentedReduceResult
ReferenceU32(const ReduceOp op, const rund::kernel::u32 *const input,
             const rund::kernel::u32 *const heads,
             rund::kernel::u32 *const output,
             const rund::kernel::u64 count) noexcept {
  switch (op) {
  case ReduceOp::Sum:
    return rund::kernel::ReferenceSegmentedReduceSumU32(input, heads, output,
                                                        count);
  case ReduceOp::CountNonzero:
    return rund::kernel::ReferenceSegmentedReduceCountNonzeroU32(input, heads,
                                                                 output, count);
  case ReduceOp::Min:
    return rund::kernel::ReferenceSegmentedReduceMinU32(input, heads, output,
                                                        count);
  case ReduceOp::Max:
    return rund::kernel::ReferenceSegmentedReduceMaxU32(input, heads, output,
                                                        count);
  }
  return SegmentedReduceResult{.element_count = count,
                               .reason =
                                   "compute_segmented_reduce_op_unsupported"};
}

[[nodiscard]] SegmentedReduceResult
ReferenceU64(const ReduceOp op, const rund::kernel::u64 *const input,
             const rund::kernel::u32 *const heads,
             rund::kernel::u64 *const output,
             const rund::kernel::u64 count) noexcept {
  switch (op) {
  case ReduceOp::Sum:
    return rund::kernel::ReferenceSegmentedReduceSumU64(input, heads, output,
                                                        count);
  case ReduceOp::CountNonzero:
    return rund::kernel::ReferenceSegmentedReduceCountNonzeroU64(input, heads,
                                                                 output, count);
  case ReduceOp::Min:
    return rund::kernel::ReferenceSegmentedReduceMinU64(input, heads, output,
                                                        count);
  case ReduceOp::Max:
    return rund::kernel::ReferenceSegmentedReduceMaxU64(input, heads, output,
                                                        count);
  }
  return SegmentedReduceResult{.element_count = count,
                               .reason =
                                   "compute_segmented_reduce_op_unsupported"};
}

} // namespace

rund::AccelCheck
ExecuteCpuSegmentedReduce(const rund::AccelDevice &pick,
                          const rund::kernel::SegmentedReduceDesc &desc,
                          const rund::kernel::SegmentedReducePlan &plan,
                          const rund::kernel::ComputeDomain domain,
                          const SegmentedReduceBinds &bindings) {
  if (!pick.check.ok || !SegmentedReduceShapeOk(desc, plan, bindings)) {
    return rund::AccelCheck{false, "compute_segmented_reduce_invalid"};
  }
  CpuBufferResult input{}, heads{}, output{};
  CpuResidentReq reqs[] = {
      CpuResidentReq{.ref = bindings.input,
                     .handle = bindings.input_handle,
                     .usage = rund::kernel::kResidentUsageRead,
                     .out = &input},
      CpuResidentReq{.ref = bindings.heads,
                     .handle = bindings.heads_handle,
                     .usage = rund::kernel::kResidentUsageRead,
                     .out = &heads},
      CpuResidentReq{.ref = bindings.output,
                     .handle = bindings.output_handle,
                     .usage = rund::kernel::kResidentUsageWrite,
                     .out = &output},
  };
  LookupCpuResidentBatch(pick, reqs);
  CpuAdapter *const adapter = CpuAdapterFromPick(pick);
  if (!input.check.ok || !heads.check.ok || !output.check.ok ||
      adapter == nullptr) {
    return rund::AccelCheck{false, "compute_segmented_reduce_invalid"};
  }

  const auto *const head_data =
      reinterpret_cast<const rund::kernel::u32 *>(heads.buffer->data.data());
  SegmentedReduceResult result{};
  if (IsSignedDomain(domain) &&
      plan.element == rund::kernel::ReduceElement::U32) {
    result = rund::kernel::ReferenceSignedSegmentedReduce(
        reinterpret_cast<const rund::kernel::i32 *>(input.buffer->data.data()),
        head_data,
        reinterpret_cast<rund::kernel::i32 *>(output.buffer->data.data()),
        plan.element_count, desc.op);
  } else if (IsSignedDomain(domain)) {
    result = rund::kernel::ReferenceSignedSegmentedReduce(
        reinterpret_cast<const rund::kernel::i64 *>(input.buffer->data.data()),
        head_data,
        reinterpret_cast<rund::kernel::i64 *>(output.buffer->data.data()),
        plan.element_count, desc.op);
  } else if (plan.element == rund::kernel::ReduceElement::U32) {
    result = ReferenceU32(
        desc.op,
        reinterpret_cast<const rund::kernel::u32 *>(input.buffer->data.data()),
        head_data,
        reinterpret_cast<rund::kernel::u32 *>(output.buffer->data.data()),
        plan.element_count);
  } else {
    result = ReferenceU64(
        desc.op,
        reinterpret_cast<const rund::kernel::u64 *>(input.buffer->data.data()),
        head_data,
        reinterpret_cast<rund::kernel::u64 *>(output.buffer->data.data()),
        plan.element_count);
  }
  if (!result.ok) {
    return rund::AccelCheck{false, result.reason};
  }
  RecordCpuDispatches(*adapter, plan.pass_count);
  return rund::AccelCheck{true, "ok"};
}

} // namespace rund::node::accel::detail
