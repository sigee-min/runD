#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../kernel/reset/projection.hpp"
#include "../../kernel/reset/proof.hpp"
#include "../../kernel/reset/stats.hpp"
#include "../pipeline/named.hpp"
#include "../resident.hpp"
#include "../scratch.hpp"
#include "local.hpp"

#include <rund/counter.hpp>

#include <limits>
#include <new>
#include <optional>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] std::string ResetSource() {
  return R"MSL(
#include <metal_stdlib>
using namespace metal;
struct ResetParams {
  ulong count;
  ulong base;
  ulong offset_words;
  ulong stride_words;
  uint element_words;
  uint reserved;
};
kernel void rund_compute_reset(device uint *target [[buffer(0)]],
                               constant ResetParams &params [[buffer(1)]],
                               uint gid [[thread_position_in_grid]]) {
  const ulong ordinal = params.base + ulong(gid);
  if (ordinal >= params.count) { return; }
  const ulong word = params.offset_words + ordinal * params.stride_words;
  target[word] = 0u;
  if (params.element_words == 2u) { target[word + 1u] = 0u; }
}
)MSL";
}

[[nodiscard]] bool PrepareResets(const rund::AccelDevice &pick,
                                 MetalAdapter &adapter,
                                 const BoundResets *const resets,
                                 MetalKernelResources &resources) {
  if (resets == nullptr || resets->size() == 0u) {
    return true;
  }
  resources.resets.reserve(static_cast<std::size_t>(resets->size()));
  for (std::uint64_t index = 0u; index < resets->size(); ++index) {
    const BoundReset &route = (*resets)[static_cast<std::size_t>(index)];
    const rund::kernel::ResidentBufferRef &ref = route.ref;
    const MetalViewTransfer *replacement = nullptr;
    if (route.external && !reset::Find(resources, route.binding, replacement)) {
      return false;
    }
    MetalResidentBufferResult resident =
        replacement == nullptr
            ? LookupMetalResidentBuffer(pick, ref, route.handle)
            : replacement->dense;
    if (!resident.check.ok || resident.device_buffer == nullptr) {
      return false;
    }
    const reset::Replacement dense{
        .count = replacement == nullptr ? 0u : replacement->count,
        .element = replacement == nullptr ? 0u : replacement->element_bytes,
    };
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)resident.device_buffer.get();
    const reset::Result proved = reset::Prove(
        reset::Project(ref, replacement == nullptr ? nullptr : &dense),
        static_cast<std::uint64_t>(buffer.length),
        std::numeric_limits<std::uint64_t>::max());
    if (!proved.check.ok) {
      return false;
    }
    resources.resets.push_back(MetalReset{
        .resident = std::move(resident),
        .range = proved.range,
    });
    constexpr std::uint64_t window = std::numeric_limits<std::uint32_t>::max();
    resources.reset_count = ::rund::detail::counter::SaturatingAdd(
        resources.reset_count, reset::Commands(proved.range.count(), window));
    resources.reset_bytes = ::rund::detail::counter::SaturatingAdd(
        resources.reset_bytes, reset::Payload(proved.range));
  }
  resources.reset_pipeline = LookupMetalNamedPipeline(adapter, "compute.reset");
  if (resources.reset_pipeline == nullptr) {
    std::shared_ptr<void> library = AcquireMetalLibrary(adapter, ResetSource());
    id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
    id<MTLLibrary> native = (__bridge id<MTLLibrary>)library.get();
    const std::uint64_t begin = MonotonicNanoseconds();
    if (!MakeNamedMetalPipeline(device, native, "rund_compute_reset",
                                resources.reset_pipeline)) {
      return false;
    }
    StoreMetalNamedPipeline(adapter, "compute.reset", resources.reset_pipeline,
                            MonotonicNanoseconds() - begin);
  }
  return resources.reset_pipeline != nullptr;
}

void CompleteMetalPrepared(void *const raw, KernelResult submitted) noexcept {
  auto *const state =
      static_cast<submission::State<MetalKernelResources> *>(raw);
  if (state == nullptr) {
    return;
  }
  const submission::Claim<MetalKernelResources> claim =
      submission::Take(*state);
  if (!claim) {
    return;
  }
  MetalKernelResources &resources = *claim.owner;
  if (submitted.check.ok && resources.adapter != nullptr) {
    submitted.check =
        FinishMetalSteps(*resources.adapter, resources, &submitted.stats);
  }
  submitted.stats.dispatch_count =
      submitted.check.ok ? resources.dispatch_count : 0u;
  SetResetStats(submitted.stats, submitted.check.ok, resources.reset_count,
                resources.reset_bytes);
  claim.completion(claim.user, submitted);
}

} // namespace

rund::AccelCheck PrepareMetalResources(
    const rund::AccelDevice &pick, const BoundStep *const steps,
    const std::size_t step_count, const std::uint64_t dispatch_count,
    const KernelPreparationMode mode, const BoundResets *const resets,
    const KernelViewLayout *const views, const RunBinds *const view_binds,
    const KernelScratchLayout *const scratch, std::uint32_t *const failed_node,
    std::shared_ptr<void> &prepared, PreparedMemory &memory) {
  @autoreleasepool {
    prepared.reset();
    memory = {};
    if (steps == nullptr || step_count == 0u) {
      return rund::AccelCheck{false, "accel_metal_unavailable"};
    }
    MetalKernelContext context{};
    const rund::AccelCheck valid = ValidateMetalKernelContext(pick, context);
    if (!valid.ok) {
      return valid;
    }
    std::shared_ptr<MetalKernelResources> resources{};
    try {
      resources = std::make_shared<MetalKernelResources>();
    } catch (const std::bad_alloc &) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    std::optional<MetalScratch> scratch_arena;
    if (scratch != nullptr) {
      if (view_binds == nullptr) {
        return rund::AccelCheck{false, "accel_kernel_scratch_invalid"};
      }
      scratch_arena.emplace(pick, *scratch, *view_binds);
      if (!scratch_arena->valid()) {
        return rund::AccelCheck{false, "accel_kernel_scratch_invalid"};
      }
    }
    rund::AccelCheck ready{};
    try {
      MetalScratchScope scratch_scope{
          scratch_arena.has_value() ? &scratch_arena.value() : nullptr};
      ready = PrepareMetalSteps(pick, steps, step_count, mode, views,
                                view_binds, failed_node, *resources);
    } catch (const std::bad_alloc &) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    if (!ready.ok) {
      return ready;
    }
    if (!PrepareResets(pick, *context.adapter, resets, *resources)) {
      return rund::AccelCheck{false, "accel_kernel_reset_invalid"};
    }
    resources->adapter = context.adapter;
    std::uint64_t physical_dispatch_count = dispatch_count;
    for (std::size_t index = 0u; index < resources->size(); ++index) {
      const MetalKernelEntry *const entry = resources->entry(index);
      const std::uint64_t auxiliary =
          entry == nullptr ? 0u : MetalViewDispatchCount(entry->view);
      if (auxiliary >
          std::numeric_limits<std::uint64_t>::max() - physical_dispatch_count) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      physical_dispatch_count += auxiliary;
    }
    if (resources->reset_count >
        std::numeric_limits<std::uint64_t>::max() - physical_dispatch_count) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    physical_dispatch_count += resources->reset_count;
    resources->dispatch_count = physical_dispatch_count;
    resources->mode = mode;
    resources->shared_scratch =
        scratch_arena.has_value() && scratch_arena->used();
    memory = resources->memory;
    prepared = std::move(resources);
    return rund::AccelCheck{true, "ok"};
  }
}

rund::AccelCheck RunMetalResources(const rund::AccelDevice &pick,
                                   const std::shared_ptr<void> &prepared) {
  @autoreleasepool {
    auto *const resources = static_cast<MetalKernelResources *>(prepared.get());
    if (resources == nullptr || resources->size() == 0u) {
      return rund::AccelCheck{false, "accel_metal_unavailable"};
    }
    MetalKernelContext context{};
    const rund::AccelCheck valid = ValidateMetalKernelContext(pick, context);
    if (!valid.ok) {
      return valid;
    }
    CommandRun command{};
    const rund::AccelCheck ready =
        OpenCommand<ResourceRefs::Borrowed>(*context.adapter, command);
    if (!ready.ok) {
      return ready;
    }
    const rund::AccelCheck encoded =
        EncodeMetalSteps(*context.adapter, *resources, command);
    if (!encoded.ok) {
      return encoded;
    }
    const rund::AccelCheck submitted =
        WaitCommand(*context.adapter, (__bridge void *)command.buffer);
    return submitted.ok ? FinishMetalSteps(*context.adapter, *resources)
                        : submitted;
  }
}

rund::AccelCheck SubmitMetalResources(const rund::AccelDevice &pick,
                                      const std::shared_ptr<void> &prepared,
                                      const KernelCompletion completion,
                                      void *const user) noexcept {
  @autoreleasepool {
    auto *const resources = static_cast<MetalKernelResources *>(prepared.get());
    if (resources == nullptr || resources->size() == 0u ||
        completion == nullptr) {
      return rund::AccelCheck{false, "accel_metal_unavailable"};
    }
    MetalKernelContext context{};
    const rund::AccelCheck valid = ValidateMetalKernelContext(pick, context);
    if (!valid.ok) {
      return valid;
    }
    CommandRun command{};
    const rund::AccelCheck ready =
        OpenCommand<ResourceRefs::Borrowed>(*context.adapter, command);
    if (!ready.ok) {
      return ready;
    }
    const rund::AccelCheck encoded =
        EncodeMetalSteps(*context.adapter, *resources, command);
    if (!encoded.ok) {
      return encoded;
    }
    submission::State<MetalKernelResources> &state = resources->submission;
    if (!submission::Begin(state, *resources, completion, user)) {
      return rund::AccelCheck{false, "compute_job_busy"};
    }
    const rund::AccelCheck submitted =
        QueueCommand(*context.adapter, (__bridge void *)command.buffer,
                     CompleteMetalPrepared, &state);
    if (!submitted.ok) {
      submission::Cancel(state);
    }
    return submitted;
  }
}
#else
rund::AccelCheck
PrepareMetalResources(const rund::AccelDevice &, const BoundStep *, std::size_t,
                      std::uint64_t, KernelPreparationMode, const BoundResets *,
                      const KernelViewLayout *, const RunBinds *,
                      const KernelScratchLayout *, std::uint32_t *,
                      std::shared_ptr<void> &, PreparedMemory &memory) {
  memory = {};
  return rund::AccelCheck{false, "accel_metal_unavailable"};
}
rund::AccelCheck RunMetalResources(const rund::AccelDevice &,
                                   const std::shared_ptr<void> &) {
  return rund::AccelCheck{false, "accel_metal_unavailable"};
}
rund::AccelCheck SubmitMetalResources(const rund::AccelDevice &,
                                      const std::shared_ptr<void> &,
                                      KernelCompletion, void *) noexcept {
  return rund::AccelCheck{false, "accel_metal_unavailable"};
}
#endif

std::uint64_t
MetalKernelTraffic(const std::shared_ptr<void> &prepared) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  const auto *const resources =
      static_cast<const MetalKernelResources *>(prepared.get());
  return resources == nullptr ? 0u : resources->traffic;
#else
  (void)prepared;
  return 0u;
#endif
}

} // namespace rund::node::accel::detail
