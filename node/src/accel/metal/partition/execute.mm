#include <accel/check.hpp>
#include <accel/device.hpp>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include "../../partition/shape.hpp"
#include "../command/run.hpp"
#include "encode/scatter.hpp"
#include "local.hpp"
#include "pipeline/select.hpp"
#include "pipeline/store.hpp"
#include "resources/pipeline.hpp"
#include "../pipeline/template.hpp"

#include <utility>

namespace rund::node::accel::detail {

void DestroyMetalPartitionEncodeResources(void *const raw) {
  auto *const resources = static_cast<MetalPartitionEncodeResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  if (resources->adapter != nullptr) {
    MetalAdapter &adapter = *resources->adapter;
    ReleaseMetalBuffer(adapter, std::move(resources->false_bits));
    ReleaseMetalBuffer(adapter, std::move(resources->false_offsets));
    ReleaseMetalBuffer(adapter, std::move(resources->false_totals));
    ReleaseMetalBuffer(adapter, std::move(resources->false_status));
  }
  delete resources;
}

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] bool
CompileMetalPartitionPipeline(MetalAdapter &adapter, const char *const key,
                              const char *const function_name,
                              std::shared_ptr<void> &out) {
  if (LookupMetalPartitionPipeline(adapter, key, out)) {
    return true;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  if (device == nil) {
    return false;
  }
  const std::uint64_t create_begin = MonotonicNanoseconds();
  if (!CompileMetalPartitionPipelineLibrary(adapter, function_name, out)) {
    return false;
  }
  StoreMetalPartitionPipeline(adapter, key, out,
                              MonotonicNanoseconds() - create_begin);
  return true;
}
#endif

bool CompileMetalPartitionPipelines(MetalAdapter &adapter,
                                    const rund::kernel::u64 flag_bytes,
                                    const rund::kernel::u64 value_bytes,
                                    MetalPartitionPipelines &out) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  const bool wide_flags = flag_bytes == sizeof(rund::kernel::u64);
  const bool wide_values = value_bytes == sizeof(rund::kernel::u64);
  const PartitionPipelineNames names =
      SelectPartitionPipelines(wide_flags, wide_values);
  return CompileMetalPartitionPipeline(adapter, names.classify_key,
                                       names.classify_function, out.classify) &&
         CompileMetalPartitionPipeline(adapter, names.scatter_key,
                                       names.scatter_function, out.scatter);
#else
  (void)adapter;
  (void)flag_bytes;
  (void)value_bytes;
  (void)out;
  return false;
#endif
}

rund::AccelCheck PrepareMetalPartition(const rund::AccelDevice &pick,
                                       const rund::kernel::PartitionDesc &desc,
                                       const rund::kernel::PartitionPlan &plan,
                                       const PartitionBinds &bindings,
                                       std::shared_ptr<void> &resources,
                                       const MetalKernelImmutablePipelines *const
                                           pipelines) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  resources.reset();
  if (!MetalPickOwnsAdapter(pick)) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  auto *const adapter = static_cast<MetalAdapter *>(pick.backend.context);
  if (adapter == nullptr || adapter->device == nullptr) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  SetMetalLastError(*adapter, "ok");
  if (!PartitionShapeOk(desc, plan, bindings)) {
    SetMetalLastError(*adapter, "compute_partition_invalid");
    return rund::AccelCheck{false, "compute_partition_invalid"};
  }

  auto *const raw = new MetalPartitionEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyMetalPartitionEncodeResources};
  raw->adapter = adapter;
  raw->plan = plan;
  rund::AccelCheck check = PlanMetalPartitionScan(*adapter, plan, *raw);
  if (!check.ok) {
    return check;
  }
  check = LookupMetalPartitionResidentBuffers(pick, bindings, *raw);
  if (!check.ok) {
    return check;
  }
  check = AcquireMetalPartitionBuffers(*adapter, plan, *raw);
  if (!check.ok) {
    return check;
  }
  if (pipelines != nullptr && pipelines->ready(5u)) {
    raw->pipelines.classify = pipelines->stages[0u];
    raw->pipelines.scatter = pipelines->stages[1u];
    raw->scan_block = pipelines->stages[2u];
    raw->scan_prefix = pipelines->stages[3u];
    raw->scan_offset = pipelines->stages[4u];
  } else if (pipelines != nullptr) {
    check = {false, "accel_metal_pipeline_unavailable"};
  } else {
    check = LoadMetalPartitionPipelines(*adapter, *raw);
  }
  if (!check.ok) {
    return check;
  }

  resources = std::move(owned);
  return rund::AccelCheck{true, "ok"};
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)bindings;
  (void)resources;
  (void)pipelines;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck EncodeMetalPartition(MetalAdapter &adapter,
                                      const std::shared_ptr<void> &resources,
                                      void *const command_encoder) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  MetalPartitionCommandState state{};
  const rund::AccelCheck prepared = PrepareMetalPartitionCommandState(
      adapter, resources, command_encoder, state);
  if (!prepared.ok) {
    return prepared;
  }
  const PartitionParams params{state.partition->plan.element_count};
  EncodeMetalPartitionClassify(state, params);
  const rund::AccelCheck scan =
      EncodeMetalPartitionScans(adapter, state, command_encoder);
  if (!scan.ok) {
    return scan;
  }
  EncodeMetalPartitionScatter(state, params);
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_encoder;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck ExecuteMetalPartition(const rund::AccelDevice &pick,
                                       const rund::kernel::PartitionDesc &desc,
                                       const rund::kernel::PartitionPlan &plan,
                                       const PartitionBinds &bindings) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const adapter = static_cast<MetalAdapter *>(pick.backend.context);
  if (!MetalPickOwnsAdapter(pick) || adapter == nullptr ||
      adapter->queue == nullptr) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  std::shared_ptr<void> resources{};
  const rund::AccelCheck prepare =
      PrepareMetalPartition(pick, desc, plan, bindings, resources);
  if (!prepare.ok) {
    return prepare;
  }
  CommandRun command{};
  const rund::AccelCheck open = OpenCommand(*adapter, command);
  if (!open.ok) {
    return open;
  }
  const rund::AccelCheck encode = EncodeMetalPartition(
      *adapter, resources, (__bridge void *)command.encoder);
  const rund::AccelCheck submit = FinishCommand(*adapter, command, encode);
  if (!submit.ok) {
    return submit;
  }
  return FinishMetalPartition(*adapter, resources);
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)bindings;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
