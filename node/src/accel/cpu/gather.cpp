#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../gather/shape.hpp"
#include "buffer/batch.hpp"

#include <kernel/program/compute/gather/reference.hpp>

#include <cstring>

namespace rund::node::accel::detail {

rund::AccelCheck ExecuteCpuGather(const rund::AccelDevice &pick,
                                  const rund::kernel::GatherDesc &desc,
                                  const rund::kernel::GatherPlan &plan,
                                  const GatherBinds &bindings) {
  if (!pick.check.ok || !GatherShapeOk(desc, plan, bindings)) {
    return rund::AccelCheck{false, "compute_gather_invalid"};
  }
  CpuBufferResult values{};
  CpuBufferResult indices{};
  CpuBufferResult logical_count{};
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
      CpuResidentReq{.ref = bindings.logical_count,
                     .handle = bindings.logical_count_handle,
                     .usage = rund::kernel::kResidentUsageRead,
                     .out = &logical_count},
  };
  LookupCpuResidentBatch(
      pick, reqs, bindings.logical_count_handle == nullptr ? 3u : 4u);
  CpuAdapter *const adapter = CpuAdapterFromPick(pick);
  if (!values.check.ok || !indices.check.ok || !output.check.ok ||
      adapter == nullptr) {
    return rund::AccelCheck{false, "compute_gather_invalid"};
  }
  const auto *const index_data =
      reinterpret_cast<const rund::kernel::u32 *>(indices.buffer->data.data());
  rund::kernel::u64 active_count = plan.element_count;
  if (bindings.logical_count_handle != nullptr) {
    if (!logical_count.check.ok || logical_count.buffer == nullptr) {
      return rund::AccelCheck{false, "compute_gather_invalid"};
    }
    if (plan.count_source == rund::kernel::ComputeCountSource::BufferU64) {
      std::memcpy(&active_count, logical_count.buffer->data.data(),
                  sizeof(active_count));
    } else {
      rund::kernel::u32 narrow = 0u;
      std::memcpy(&narrow, logical_count.buffer->data.data(), sizeof(narrow));
      active_count = narrow;
    }
    if (active_count > plan.element_count) {
      return rund::AccelCheck{false, "compute_bounded_count_invalid"};
    }
  }
  const bool u32 = plan.element == rund::kernel::GatherElement::U32;
  const rund::kernel::GatherResult result =
      u32 ? rund::kernel::ReferenceGatherU32(
                reinterpret_cast<const rund::kernel::u32 *>(
                    values.buffer->data.data()),
                index_data,
                reinterpret_cast<rund::kernel::u32 *>(
                    output.buffer->data.data()),
                active_count, plan.source_count)
          : rund::kernel::ReferenceGatherU64(
                reinterpret_cast<const rund::kernel::u64 *>(
                    values.buffer->data.data()),
                index_data,
                reinterpret_cast<rund::kernel::u64 *>(
                    output.buffer->data.data()),
                active_count, plan.source_count);
  if (!result.ok) {
    return rund::AccelCheck{false, result.reason};
  }
  RecordCpuDispatches(*adapter, plan.pass_count);
  return rund::AccelCheck{true, "ok"};
}

} // namespace rund::node::accel::detail
