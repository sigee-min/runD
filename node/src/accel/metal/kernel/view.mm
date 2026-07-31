#include "view.hpp"

#include "../../clock.hpp"
#include "../../kernel/preparation.hpp"
#include "../buffer/owner.hpp"
#include "../buffer/resident/find.hpp"
#include "../pipeline/cache.hpp"
#include "../pipeline/named.hpp"
#include "../resident.hpp"
#include "../resident/access.hpp"
#include "../state.hpp"

#include <algorithm>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <utility>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

struct MetalViewParams final {
  std::uint64_t count{};
  std::uint64_t offset_words{};
  std::uint64_t stride_words{};
  std::uint32_t element_words{};
  std::uint32_t reserved{};
};

static_assert(sizeof(MetalViewParams) == 32u);

[[nodiscard]] std::string MetalViewSource() {
  return R"MSL(
#include <metal_stdlib>
using namespace metal;

struct ViewParams {
  ulong count;
  ulong offset_words;
  ulong stride_words;
  uint element_words;
  uint reserved;
};

kernel void rund_pipeline_view_gather(
    device const uint *source [[buffer(0)]],
    device uint *dense [[buffer(1)]],
    constant ViewParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (ulong(gid) >= params.count) { return; }
  const ulong source_word =
      params.offset_words + ulong(gid) * params.stride_words;
  const ulong dense_word = ulong(gid) * ulong(params.element_words);
  dense[dense_word] = source[source_word];
  if (params.element_words == 2u) {
    dense[dense_word + 1u] = source[source_word + 1u];
  }
}

kernel void rund_pipeline_view_scatter(
    device const uint *dense [[buffer(0)]],
    device uint *target [[buffer(1)]],
    constant ViewParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (ulong(gid) >= params.count) { return; }
  const ulong dense_word = ulong(gid) * ulong(params.element_words);
  const ulong target_word =
      params.offset_words + ulong(gid) * params.stride_words;
  target[target_word] = dense[dense_word];
  if (params.element_words == 2u) {
    target[target_word + 1u] = dense[dense_word + 1u];
  }
}
)MSL";
}

[[nodiscard]] std::shared_ptr<void>
AcquireViewPipeline(MetalAdapter &adapter, const char *const key,
                    const char *const function) {
  std::shared_ptr<void> pipeline = LookupMetalNamedPipeline(adapter, key);
  if (pipeline != nullptr) {
    return pipeline;
  }
  std::shared_ptr<void> library =
      AcquireMetalLibrary(adapter, MetalViewSource());
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  id<MTLLibrary> native_library = (__bridge id<MTLLibrary>)library.get();
  const std::uint64_t begin = MonotonicNanoseconds();
  if (!MakeNamedMetalPipeline(device, native_library, function, pipeline)) {
    return {};
  }
  StoreMetalNamedPipeline(adapter, key, pipeline,
                          MonotonicNanoseconds() - begin);
  return pipeline;
}

[[nodiscard]] bool DenseView(const rund::kernel::ResidentBufferRef &ref) {
  // Stride cannot change the selected address set for zero or one element.
  return ref.count > 1u && ref.stride_bytes != ref.element_bytes;
}

struct Replacement final {
  MetalResidentBufferResult resident{};
};

[[nodiscard]] MetalResidentBufferResult
ResolveExternal(const rund::AccelDevice &pick,
                const rund::kernel::ResidentBufferRef &ref,
                const std::shared_ptr<void> &handle) {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr) {
    return RejectResident<MetalResidentBufferResult>(
        "accel_metal_resident_owner_invalid");
  }
  MetalResidentState &resident = MetalResidents(*adapter);
  std::lock_guard lock{resident.mutex};
  MetalResidentBufferResult result = ResolveMetalResidentBuffer(
      resident, ref, handle, "accel_metal_resident_id_unavailable", true);
  if (result.check.ok) {
    result.ref = ref;
  }
  return result;
}

[[nodiscard]] MetalResidentBufferResult
ResolveDense(const rund::AccelDevice &pick, const std::uint64_t binding,
             const rund::kernel::ResidentBufferRef &requested,
             const KernelPreparationMode mode,
             const KernelViewLayout *const views,
             const RunBinds *const view_binds, bool &planned) {
  planned = views != nullptr || view_binds != nullptr;
  if (!planned) {
    if (IsPipelinePrivatePreparation(mode)) {
      return RejectResident<MetalResidentBufferResult>(
          "compute_pipeline_memory_plan_invalid");
    }
    const std::uint64_t bytes = requested.count * requested.element_bytes;
    return CreateMetalResidentBuffer(
        pick,
        ResidentDesc{.bytes = bytes,
                     .element_bytes = requested.element_bytes,
                     .stride_bytes = requested.element_bytes,
                     .count = requested.count,
                     .usage = requested.usage},
        false);
  }
  if (views == nullptr || view_binds == nullptr || !view_binds->valid()) {
    return RejectResident<MetalResidentBufferResult>(
        "compute_pipeline_memory_plan_invalid");
  }
  const std::uint64_t bytes = requested.count * requested.element_bytes;
  const KernelViewSlot *const slot =
      FindKernelViewSlot(*views, binding, bytes);
  if (slot == nullptr || slot->slot >= view_binds->size()) {
    return RejectResident<MetalResidentBufferResult>(
        "compute_pipeline_memory_plan_invalid");
  }
  rund::kernel::ResidentBufferRef ref = view_binds->refs()[slot->slot];
  if (ref.offset_bytes > ref.bytes || bytes > ref.bytes - ref.offset_bytes) {
    return RejectResident<MetalResidentBufferResult>(
        "compute_pipeline_memory_plan_invalid");
  }
  ref.element_bytes = requested.element_bytes;
  ref.stride_bytes = requested.element_bytes;
  ref.count = requested.count;
  ref.usage = requested.usage;
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr) {
    return RejectResident<MetalResidentBufferResult>(
        "accel_metal_resident_owner_invalid");
  }
  MetalResidentState &resident = MetalResidents(*adapter);
  std::lock_guard lock{resident.mutex};
  MetalResidentBufferResult result = ResolveMetalResidentBuffer(
      resident, ref, view_binds->handles()[slot->slot],
      "accel_metal_resident_id_unavailable");
  if (result.check.ok) {
    result.ref = ref;
  }
  return result;
}

[[nodiscard]] rund::AccelCheck
EncodeTransfers(const MetalViewLowering &view,
                id<MTLComputeCommandEncoder> encoder, const bool inputs) {
  if (encoder == nil) {
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  id<MTLComputePipelineState> pipeline =
      (__bridge id<MTLComputePipelineState>)(inputs
                                                 ? view.gather_pipeline.get()
                                                 : view.scatter_pipeline.get());
  bool encoded = false;
  for (const MetalViewTransfer &transfer : view.transfers) {
    if (transfer.input != inputs) {
      continue;
    }
    if (pipeline == nil || transfer.external.device_buffer == nullptr ||
        transfer.dense.device_buffer == nullptr ||
        (transfer.element_bytes != 4u && transfer.element_bytes != 8u) ||
        transfer.offset_bytes % sizeof(std::uint32_t) != 0u ||
        transfer.stride_bytes % sizeof(std::uint32_t) != 0u ||
        transfer.count > static_cast<std::uint64_t>(
                             std::numeric_limits<std::uint32_t>::max())) {
      return rund::AccelCheck{false, "compute_resident_stride_invalid"};
    }
    const MetalViewParams params{
        .count = transfer.count,
        .offset_words = transfer.offset_bytes / sizeof(std::uint32_t),
        .stride_words = transfer.stride_bytes / sizeof(std::uint32_t),
        .element_words = static_cast<std::uint32_t>(transfer.element_bytes /
                                                    sizeof(std::uint32_t)),
    };
    id<MTLBuffer> external =
        (__bridge id<MTLBuffer>)transfer.external.device_buffer.get();
    id<MTLBuffer> dense =
        (__bridge id<MTLBuffer>)transfer.dense.device_buffer.get();
    const NSUInteger dense_offset =
        static_cast<NSUInteger>(transfer.dense.ref.offset_bytes);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(inputs ? external : dense)
                offset:(inputs ? 0u : dense_offset)atIndex:0u];
    [encoder setBuffer:(inputs ? dense : external)
                offset:(inputs ? dense_offset : 0u)atIndex:1u];
    [encoder setBytes:&params length:sizeof(params) atIndex:2u];
    const NSUInteger count = static_cast<NSUInteger>(transfer.count);
    [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(count, 256u), 1u,
                                          1u)];
    encoded = true;
  }
  const bool closes_input_lifetime = !inputs && view.has_input;
  if (encoded || closes_input_lifetime) {
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }
  return rund::AccelCheck{true, "ok"};
}

} // namespace

rund::AccelCheck PrepareMetalViewLowering(
    const rund::AccelDevice &pick, const BoundStep &source,
    const KernelPreparationMode mode, const KernelViewLayout *const views,
    const RunBinds *const view_binds, std::shared_ptr<MetalViewLowering> &out) {
  out.reset();
  if (source.step == nullptr || source.source_binds == nullptr ||
      source.step->kind() == rund::kernel::NodeKind::Map ||
      source.step->kind() == rund::kernel::NodeKind::ScatterReduce) {
    return rund::AccelCheck{true, "ok"};
  }
  const RunBinds &original = *source.source_binds;
  if (!original.valid()) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  std::shared_ptr<MetalViewLowering> view;
  try {
    view = std::make_shared<MetalViewLowering>();
    view->transfers.reserve(source.step->graph_binding_indices.size());
    view->transfer_by_binding.resize(original.size(), 0u);
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::vector<Replacement> replacements;
  std::vector<std::uint32_t> replacement_by_binding;
  try {
    replacements.reserve(source.step->graph_binding_indices.size());
    replacement_by_binding.resize(original.size(), 0u);
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  for (std::size_t local = 0u;
       local < source.step->graph_binding_indices.size(); ++local) {
    const std::uint64_t index = source.step->graph_binding_indices[local];
    if (index >= original.size() ||
        replacement_by_binding[static_cast<std::size_t>(index)] != 0u) {
      continue;
    }
    const rund::kernel::ResidentBufferRef &ref = original.refs()[index];
    const bool normalize_singleton =
        ref.count == 1u && ref.stride_bytes != ref.element_bytes;
    if ((!DenseView(ref) && !normalize_singleton) || ref.element_bytes == 0u ||
        ref.count >
            std::numeric_limits<std::uint64_t>::max() / ref.element_bytes) {
      continue;
    }
    MetalResidentBufferResult external =
        ResolveExternal(pick, ref, original.handles()[index]);
    if (!external.check.ok) {
      return external.check;
    }
    if (normalize_singleton) {
      external.ref.stride_bytes = external.ref.element_bytes;
      try {
        replacements.push_back(Replacement{.resident = std::move(external)});
        replacement_by_binding[static_cast<std::size_t>(index)] =
            static_cast<std::uint32_t>(replacements.size());
      } catch (const std::bad_alloc &) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      continue;
    }
    bool planned = false;
    MetalResidentBufferResult dense = ResolveDense(
        pick, index, ref, mode, views, view_binds, planned);
    if (!dense.check.ok) {
      return dense.check;
    }
    try {
      view->transfers.push_back(MetalViewTransfer{
          .binding = index,
          .external = std::move(external),
          .dense = dense,
          .count = ref.count,
          .element_bytes = ref.element_bytes,
          .offset_bytes = ref.offset_bytes,
          .stride_bytes = ref.stride_bytes,
          .input = ref.usage == rund::kernel::kResidentUsageRead,
          .planned = planned,
      });
      view->has_input = view->has_input || view->transfers.back().input;
      replacements.push_back(Replacement{.resident = std::move(dense)});
      view->transfer_by_binding[static_cast<std::size_t>(index)] =
          static_cast<std::uint32_t>(view->transfers.size());
      replacement_by_binding[static_cast<std::size_t>(index)] =
          static_cast<std::uint32_t>(replacements.size());
    } catch (const std::bad_alloc &) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  if (replacements.empty()) {
    return rund::AccelCheck{true, "ok"};
  }
  view->binds.reserve(original.size());
  for (std::uint64_t index = 0u; index < original.size(); ++index) {
    const std::uint32_t ordinal =
        replacement_by_binding[static_cast<std::size_t>(index)];
    const Replacement *const replacement =
        ordinal == 0u ? nullptr : &replacements[ordinal - 1u];
    if (!view->binds.push(replacement == nullptr ? original.refs()[index]
                                                 : replacement->resident.ref,
                          replacement == nullptr
                              ? original.handles()[index]
                              : replacement->resident.handle)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  if (!view->binds.valid() ||
      !RebindBoundStep(source, view->binds, view->step)) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  bool needs_gather = false;
  bool needs_scatter = false;
  for (const MetalViewTransfer &transfer : view->transfers) {
    needs_gather = needs_gather || transfer.input;
    needs_scatter = needs_scatter || !transfer.input;
  }
  if (needs_gather) {
    view->gather_pipeline = AcquireViewPipeline(
        *adapter, "pipeline.view.gather.u32", "rund_pipeline_view_gather");
  }
  if (needs_scatter) {
    view->scatter_pipeline = AcquireViewPipeline(
        *adapter, "pipeline.view.scatter.u32", "rund_pipeline_view_scatter");
  }
  if ((needs_gather && view->gather_pipeline == nullptr) ||
      (needs_scatter && view->scatter_pipeline == nullptr)) {
    return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
  }
  out = std::move(view);
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck
EncodeMetalViewInputs(const std::shared_ptr<MetalViewLowering> &view,
                      id<MTLComputeCommandEncoder> encoder) {
  return view == nullptr ? rund::AccelCheck{true, "ok"}
                         : EncodeTransfers(*view, encoder, true);
}

rund::AccelCheck
EncodeMetalViewOutputs(const std::shared_ptr<MetalViewLowering> &view,
                       id<MTLComputeCommandEncoder> encoder) {
  return view == nullptr ? rund::AccelCheck{true, "ok"}
                         : EncodeTransfers(*view, encoder, false);
}

PreparedMemory MetalViewMemory(const std::shared_ptr<MetalViewLowering> &view,
                               const std::uint64_t budget,
                               std::uint64_t &traffic) noexcept {
  std::uint64_t bytes = 0u;
  traffic = 0u;
  if (view != nullptr) {
    for (const MetalViewTransfer &transfer : view->transfers) {
      const std::uint64_t dense = transfer.count * transfer.element_bytes;
      traffic = ::rund::detail::counter::SaturatingAdd(traffic, dense);
      if (!transfer.planned) {
        bytes = bytes > std::numeric_limits<std::uint64_t>::max() - dense
                    ? std::numeric_limits<std::uint64_t>::max()
                    : bytes + dense;
      }
    }
  }
  return PreparedMemory{
      .current = bytes, .peak = bytes, .cumulative = bytes, .budget = budget};
}

std::uint64_t MetalViewDispatchCount(
    const std::shared_ptr<MetalViewLowering> &view) noexcept {
  return view == nullptr ? 0u
                         : static_cast<std::uint64_t>(view->transfers.size());
}

#endif

} // namespace rund::node::accel::detail
