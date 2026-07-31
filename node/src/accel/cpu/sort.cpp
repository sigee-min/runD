#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../domain.hpp"
#include "../sort/shape.hpp"
#include "buffer/batch.hpp"
#include "sort/count.hpp"
#include "sort/radix.hpp"
#include <limits>
#include <memory>
#include <mutex>
namespace rund::node::accel::detail {
rund::AccelCheck ExecuteCpuSort(const rund::AccelDevice &pick,
                                const rund::kernel::SortDesc &desc,
                                const rund::kernel::SortPlan &plan,
                                const rund::kernel::ComputeDomain domain,
                                const SortBinds &bindings) {
  if (!pick.check.ok || !SortShapeOk(desc, plan, bindings)) {
    return rund::AccelCheck{false, "compute_sort_invalid"};
  }
  CpuBufferResult read_keys{};
  CpuBufferResult write_keys{};
  CpuBufferResult write_values{};
  CpuBufferResult read_values{};
  CpuResidentReq reqs[] = {
      CpuResidentReq{.ref = bindings.read_keys,
                     .handle = bindings.read_keys_handle,
                     .usage = rund::kernel::kResidentUsageRead,
                     .out = &read_keys},
      CpuResidentReq{.ref = bindings.write_keys,
                     .handle = bindings.write_keys_handle,
                     .usage = rund::kernel::kResidentUsageWrite,
                     .out = &write_keys},
      CpuResidentReq{.ref = bindings.write_values,
                     .handle = bindings.write_values_handle,
                     .usage = rund::kernel::kResidentUsageWrite,
                     .out = &write_values},
      CpuResidentReq{.ref = bindings.read_values,
                     .handle = bindings.read_values_handle,
                     .usage = rund::kernel::kResidentUsageRead,
                     .out = &read_values},
  };
  LookupCpuResidentBatch(pick, reqs);
  CpuAdapter *const adapter = CpuAdapterFromPick(pick);
  if (!read_keys.check.ok || !write_keys.check.ok || !write_values.check.ok ||
      adapter == nullptr) {
    return rund::AccelCheck{false, "compute_sort_invalid"};
  }
  rund::kernel::u64 active_count = plan.element_count;
  const rund::AccelCheck count =
      ReadCpuSortCount(pick, plan, bindings, active_count);
  if (!count.ok) {
    return count;
  }
  if (plan.element_count >
      static_cast<rund::kernel::u64>(std::numeric_limits<std::size_t>::max())) {
    return rund::AccelCheck{false, "compute_sort_invalid"};
  }

  const bool identity_values =
      plan.value == rund::kernel::SortValue::IdentityU32;
  const bool signed_order = IsSignedDomain(domain);
  const rund::kernel::u32 *values = nullptr;
  if (!identity_values) {
    if (!read_values.check.ok) {
      return rund::AccelCheck{false, "compute_sort_invalid"};
    }
    values = reinterpret_cast<const rund::kernel::u32 *>(
        read_values.buffer->data.data());
  }
  rund::AccelCheck result{};
  {
    std::lock_guard<std::mutex> lock{adapter->mutex};
    if (adapter->sort_scratch == nullptr) {
      adapter->sort_scratch = std::make_shared<CpuSortScratch>();
    }
    CpuSortScratch &scratch = *adapter->sort_scratch;
    const bool u32 = plan.key == rund::kernel::SortKey::U32;
    result =
        u32 ? ExecuteCpuRadixSort(reinterpret_cast<const rund::kernel::u32 *>(
                                      read_keys.buffer->data.data()),
                                  values,
                                  reinterpret_cast<rund::kernel::u32 *>(
                                      write_keys.buffer->data.data()),
                                  reinterpret_cast<rund::kernel::u32 *>(
                                      write_values.buffer->data.data()),
                                  active_count, plan.radix_pass_count,
                                  identity_values, signed_order, scratch)
            : ExecuteCpuRadixSort(reinterpret_cast<const rund::kernel::u64 *>(
                                      read_keys.buffer->data.data()),
                                  values,
                                  reinterpret_cast<rund::kernel::u64 *>(
                                      write_keys.buffer->data.data()),
                                  reinterpret_cast<rund::kernel::u32 *>(
                                      write_values.buffer->data.data()),
                                  active_count, plan.radix_pass_count,
                                  identity_values, signed_order, scratch);
  }
  if (!result.ok) {
    return result;
  }
  RecordCpuDispatches(*adapter, plan.radix_pass_count * 3u);
  return rund::AccelCheck{true, "ok"};
}
} // namespace rund::node::accel::detail
