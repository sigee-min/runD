#include "model.hpp"

#include "../../../scatter/reduce/model.hpp"

#include "../../adapter.hpp"
#include "../../buffer/owner.hpp"
#include "../../buffer/resident/batch.hpp"

#include <cstdint>
#include <memory>
#include <utility>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

void DestroyMetalScatterReduce(void *const raw) {
  auto *const resources = static_cast<MetalScatterReduceResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  if (resources->adapter != nullptr) {
    ReleaseMetalBuffer(*resources->adapter, std::move(resources->status));
    ReleaseMetalBuffer(*resources->adapter, std::move(resources->indirect));
    ReleaseMetalBuffer(*resources->adapter, std::move(resources->counts));
  }
  delete resources;
}

#endif

rund::AccelCheck PrepareMetalScatterReduce(
    const rund::AccelDevice &pick, const rund::kernel::ScatterReducePlan &plan,
    const ScatterReduceBinds &bindings, std::shared_ptr<void> &resources) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  resources.reset();
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr || adapter->device == nullptr || !plan.ok) {
    return {false, "compute_scatter_reduce_buffer_invalid"};
  }
  auto *const raw = new MetalScatterReduceResources{};
  std::shared_ptr<void> owned{raw, DestroyMetalScatterReduce};
  raw->adapter = adapter;
  raw->plan = plan;
  MetalResidentReq base[] = {
      {bindings.values, bindings.values_handle, &raw->values},
      {bindings.indices, bindings.indices_handle, &raw->indices},
      {bindings.output, bindings.output_handle, &raw->output},
  };
  LookupMetalResidentBatch(pick, base, "accel_metal_resident_id_unavailable");
  if (!raw->values.check.ok || !raw->indices.check.ok ||
      !raw->output.check.ok || raw->values.device_buffer == nullptr ||
      raw->indices.device_buffer == nullptr ||
      raw->output.device_buffer == nullptr) {
    return {false, "compute_scatter_reduce_buffer_invalid"};
  }
  if (plan.count_source != rund::kernel::ComputeCountSource::Descriptor) {
    MetalResidentReq count[] = {
        {bindings.count, bindings.count_handle, &raw->count}};
    LookupMetalResidentBatch(pick, count,
                             "accel_metal_resident_id_unavailable");
    if (!raw->count.check.ok || raw->count.device_buffer == nullptr) {
      return {false, "compute_scatter_reduce_buffer_invalid"};
    }
  }
  raw->status =
      AcquireMetalBuffer(*adapter, plan.status_bytes, MetalBufferUsage::Output);
  raw->indirect = AcquireMetalBuffer(*adapter, plan.indirect_bytes,
                                     MetalBufferUsage::Output);
  raw->counts =
      AcquireMetalBuffer(*adapter, plan.output_count * sizeof(std::uint32_t),
                         MetalBufferUsage::Scratch);
  if (raw->status.buffer == nullptr || raw->indirect.buffer == nullptr ||
      raw->counts.buffer == nullptr ||
      !AcquireMetalScatterReducePipelines(*adapter, plan, raw->control_pipeline,
                                          raw->init_pipeline,
                                          raw->fold_pipeline)) {
    return {false, "accel_metal_pipeline_unavailable"};
  }
  resources = std::move(owned);
  return {true, "ok"};
#else
  (void)pick;
  (void)plan;
  (void)bindings;
  (void)resources;
  return {false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
