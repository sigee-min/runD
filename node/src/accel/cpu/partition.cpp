#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../partition/shape.hpp"
#include "buffer/batch.hpp"
#include "partition/run.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck
ExecuteCpuPartition(const rund::AccelDevice &pick,
                    const rund::kernel::PartitionDesc &desc,
                    const rund::kernel::PartitionPlan &plan,
                    const PartitionBinds &bindings) {
  if (!pick.check.ok || !PartitionShapeOk(desc, plan, bindings)) {
    return rund::AccelCheck{false, "compute_partition_invalid"};
  }
  CpuBufferResult flags{};
  CpuBufferResult values{};
  CpuBufferResult output{};
  CpuResidentReq reqs[] = {
      CpuResidentReq{.ref = bindings.flags,
                     .handle = bindings.flags_handle,
                     .usage = rund::kernel::kResidentUsageRead,
                     .out = &flags},
      CpuResidentReq{.ref = bindings.values,
                     .handle = bindings.values_handle,
                     .usage = rund::kernel::kResidentUsageRead,
                     .out = &values},
      CpuResidentReq{.ref = bindings.output,
                     .handle = bindings.output_handle,
                     .usage = rund::kernel::kResidentUsageWrite,
                     .out = &output},
  };
  LookupCpuResidentBatch(pick, reqs);
  CpuAdapter *const adapter = CpuAdapterFromPick(pick);
  if (!flags.check.ok || !values.check.ok || !output.check.ok ||
      adapter == nullptr) {
    return rund::AccelCheck{false, "compute_partition_invalid"};
  }
  rund::kernel::u64 false_count = 0u;
  rund::kernel::u64 true_count = 0u;
  const rund::kernel::PartitionResult result =
      RunCpuPartition(plan, flags, values, output, false_count, true_count);
  if (!result.ok) {
    return rund::AccelCheck{false, result.reason};
  }
  RecordCpuDispatches(*adapter, plan.pass_count);
  return rund::AccelCheck{true, "ok"};
}

} // namespace rund::node::accel::detail
