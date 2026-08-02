#include <accel/check.hpp>
#include <accel/device.hpp>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include "../../reduce/metal.hpp"
#include "../../reduce/shape.hpp"
#include "../command/run.hpp"
#include "encode/pass.hpp"
#include "local.hpp"
#include "pipeline/store.hpp"
#include "resources/pipeline.hpp"
#include "../pipeline/template.hpp"

#include <utility>

namespace rund::node::accel::detail {

void DestroyMetalReduceEncodeResources(void *const raw) {
  auto *const resources = static_cast<MetalReduceEncodeResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  if (resources->adapter != nullptr) {
    ReleaseMetalBuffer(*resources->adapter, std::move(resources->partial));
    ReleaseMetalBuffer(*resources->adapter, std::move(resources->status));
  }
  delete resources;
}

bool CompileMetalReducePipeline(MetalAdapter &adapter,
                                const rund::kernel::ReducePlan &plan,
                                const rund::kernel::ComputeDomain domain,
                                std::shared_ptr<void> &out) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  const std::string key = ReducePipelineKey(plan, domain);
  if (LookupMetalReducePipeline(adapter, key, out)) {
    return true;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  if (device == nil) {
    return false;
  }
  const std::uint64_t create_begin = MonotonicNanoseconds();
  if (!CompileMetalReducePipelineLibrary(adapter, plan, domain, out)) {
    return false;
  }
  StoreMetalReducePipeline(adapter, key, out,
                           MonotonicNanoseconds() - create_begin);
  return true;
#else
  (void)adapter;
  (void)plan;
  (void)domain;
  (void)out;
  return false;
#endif
}

rund::AccelCheck PrepareMetalReduce(const rund::AccelDevice &pick,
                                    const rund::kernel::ReduceDesc &desc,
                                    const rund::kernel::ReducePlan &plan,
                                    const rund::kernel::ComputeDomain domain,
                                    const ReduceBinds &bindings,
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
  if (!ReduceShapeOk(desc, plan, bindings)) {
    SetMetalLastError(*adapter, "compute_reduce_invalid");
    return rund::AccelCheck{false, "compute_reduce_invalid"};
  }

  auto *const raw = new MetalReduceEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyMetalReduceEncodeResources};
  raw->adapter = adapter;
  raw->plan = plan;
  rund::AccelCheck check =
      LookupMetalReduceResidentBuffers(pick, bindings, *raw);
  if (check.ok) {
    check = PrepareMetalReduceBuffers(*adapter, plan, *raw);
  }
  if (check.ok) {
    if (pipelines != nullptr && pipelines->ready(1u)) {
      raw->pipeline = pipelines->stages[0u];
    } else if (pipelines != nullptr) {
      check = {false, "accel_metal_pipeline_unavailable"};
    } else {
      check = PrepareMetalReducePipeline(*adapter, plan, domain, *raw);
    }
  }
  if (!check.ok) {
    SetMetalLastError(*adapter, check.reason);
    return check;
  }
  resources = std::move(owned);
  return rund::AccelCheck{true, "ok"};
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)domain;
  (void)bindings;
  (void)resources;
  (void)pipelines;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck EncodeMetalReduce(MetalAdapter &adapter,
                                   const std::shared_ptr<void> &resources,
                                   void *command_encoder) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  MetalReduceCommandState state{};
  const rund::AccelCheck prepared = PrepareMetalReduceCommandState(
      adapter, resources, command_encoder, state);
  if (!prepared.ok) {
    return prepared;
  }

  if (state.reduce->plan.op == rund::kernel::ReduceOp::Sum ||
      state.reduce->plan.op == rund::kernel::ReduceOp::CountNonzero) {
    const rund::kernel::u64 groups = state.reduce->plan.first_pass_group_count;
    const bool final = state.reduce->plan.pass_count == 1u;
    const ReducePassParams first{
        0u,
        0u,
        state.reduce->plan.element_count,
        groups,
        final ? 1u : 0u,
        1u,
        static_cast<rund::kernel::u32>(
            rund::kernel::ComputeCountBytes(state.reduce->plan.count_source) /
            sizeof(rund::kernel::u32))};
    EncodeMetalReducePass(state, first, false, groups);
    if (!final) {
      [state.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      const ReducePassParams finish{0u, 0u, groups, 1u, 1u, 0u, 0u};
      EncodeMetalReducePass(state, finish, true, 1u);
    }
    return rund::AccelCheck{true, "ok"};
  }

  rund::kernel::u64 current = state.reduce->plan.element_count;
  rund::kernel::u64 read_offset = 0u;
  rund::kernel::u64 write_offset = 0u;
  bool read_partial = false;
  for (rund::kernel::u64 pass = 0u; pass < state.reduce->plan.pass_count;
       ++pass) {
    const rund::kernel::u64 next =
        rund::kernel::ReduceGroupCount(current, state.reduce->plan.block_size);
    const bool final_pass = next == 1u;
    const ReducePassParams params{
        read_offset,
        write_offset,
        current,
        next,
        final_pass ? 1u : 0u,
        pass == 0u ? 1u : 0u,
        static_cast<rund::kernel::u32>(
            pass == 0u ? rund::kernel::ComputeCountBytes(
                             state.reduce->plan.count_source) /
                             sizeof(rund::kernel::u32)
                       : 0u)};
    EncodeMetalReducePass(state, params, read_partial, next);
    if (final_pass) {
      break;
    }
    [state.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    read_partial = true;
    read_offset = write_offset;
    write_offset += next;
    current = next;
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_encoder;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck ExecuteMetalReduce(const rund::AccelDevice &pick,
                                    const rund::kernel::ReduceDesc &desc,
                                    const rund::kernel::ReducePlan &plan,
                                    const rund::kernel::ComputeDomain domain,
                                    const ReduceBinds &bindings) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const adapter = static_cast<MetalAdapter *>(pick.backend.context);
  if (!MetalPickOwnsAdapter(pick) || adapter == nullptr ||
      adapter->queue == nullptr) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  std::shared_ptr<void> resources{};
  const rund::AccelCheck prepare =
      PrepareMetalReduce(pick, desc, plan, domain, bindings, resources);
  if (!prepare.ok) {
    return prepare;
  }

  CommandRun command{};
  const rund::AccelCheck open = OpenCommand(*adapter, command);
  if (!open.ok) {
    return open;
  }
  const rund::AccelCheck encode =
      EncodeMetalReduce(*adapter, resources, (__bridge void *)command.encoder);
  const rund::AccelCheck submit = FinishCommand(*adapter, command, encode);
  if (!submit.ok) {
    return submit;
  }
  return FinishMetalReduce(*adapter, resources);
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)domain;
  (void)bindings;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
