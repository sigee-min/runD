#pragma once

#include "../../../clock.hpp"
#include "../../../kernel/backend/run.hpp"
#include "../../pipeline/named.hpp"
#include "resources.hpp"

#include <kernel/program/compute/lowering/text.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

struct MetalMapControlWindow final {
  std::uint64_t begin{};
  std::uint64_t count{};
};

struct MetalMapControlConfig final {
  std::uint32_t has_count{};
  std::uint32_t count_u64{};
  std::uint32_t has_predicate{};
  std::uint32_t predicate_u64{};
  std::uint32_t dispatch_width{};
  std::uint32_t checked{};
  std::uint64_t capacity{};
  std::uint64_t predicate_expected{};
};

static_assert(sizeof(MetalMapControlWindow) == 16u);
static_assert(sizeof(MetalMapControlConfig) == 40u);

[[nodiscard]] inline rund::kernel::LoweringArtifact
MetalControlledMapArtifact(const rund::kernel::LoweringArtifact &source,
                           const rund::kernel::ComputePlan &plan) {
  rund::kernel::LoweringArtifact artifact = source;
  const std::uint64_t binding =
      plan.input_buffer_count + plan.output_buffer_count + 1u;
  const std::string entry =
      "    uint gid [[thread_position_in_grid]]) {\n";
  const std::string controlled =
      "    const device uint* rund_control_active [[buffer(" +
      std::to_string(binding) +
      ")]],\n    uint gid [[thread_position_in_grid]]) {\n"
      "  if (gid >= rund_control_active[0]) { return; }\n";
  const std::size_t at = artifact.source_text.find(entry);
  std::string function = "rund_compute_map_";
  rund::kernel::compute_lowering_detail::AppendHex64Digits(
      function, artifact.key.op_hash_hi);
  function += "_";
  rund::kernel::compute_lowering_detail::AppendHex64Digits(
      function, artifact.key.op_hash_lo);
  const std::size_t function_at = artifact.source_text.find(function);
  const std::string variant = "// artifact_variant=canonical";
  const std::size_t variant_at = artifact.source_text.find(variant);
  if (at == std::string::npos ||
      artifact.source_text.find(entry, at + entry.size()) !=
          std::string::npos ||
      function_at == std::string::npos ||
      artifact.source_text.find(function, function_at + function.size()) !=
          std::string::npos ||
      variant_at == std::string::npos ||
      artifact.source_text.find(variant, variant_at + variant.size()) !=
          std::string::npos) {
    artifact.ok = false;
    artifact.reason = "compute_artifact_mismatch";
    return artifact;
  }
  artifact.source_text.replace(at, entry.size(), controlled);
  artifact.source_text.replace(function_at, function.size(),
                               function + "_controlled");
  artifact.source_text.replace(variant_at, variant.size(),
                               "// artifact_variant=controlled");
  artifact.key.variant =
      rund::kernel::LoweringArtifactVariant::Controlled;
  return artifact;
}

[[nodiscard]] inline std::string MetalMapControlSource() {
  return R"metal(
#include <metal_stdlib>
using namespace metal;

struct ControlWindow {
  ulong begin;
  ulong count;
};

struct ControlConfig {
  uint has_count;
  uint count_u64;
  uint has_predicate;
  uint predicate_u64;
  uint dispatch_width;
  uint checked;
  ulong capacity;
  ulong predicate_expected;
};

kernel void rund_map_control_dispatch(
    device const uchar *count_source [[buffer(0)]],
    device const uchar *predicate_source [[buffer(1)]],
    device uint *dispatch_args [[buffer(2)]],
    constant ControlWindow *windows [[buffer(3)]],
    constant ControlConfig &config [[buffer(4)]],
    device atomic_uint *status [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  ulong logical = config.capacity;
  if (config.has_count != 0u) {
    logical = config.count_u64 != 0u
                  ? *reinterpret_cast<device const ulong *>(count_source)
                  : ulong(*reinterpret_cast<device const uint *>(count_source));
  }
  const bool overflow = logical > config.capacity;
  const uint prior = atomic_load_explicit(&status[0], memory_order_relaxed);
  bool enabled = true;
  if (config.has_predicate != 0u) {
    const ulong observed =
        config.predicate_u64 != 0u
            ? *reinterpret_cast<device const ulong *>(predicate_source)
            : ulong(*reinterpret_cast<device const uint *>(predicate_source));
    enabled = observed == config.predicate_expected;
  }
  const ControlWindow window = windows[gid];
  const ulong remaining = !overflow && logical > window.begin
                              ? logical - window.begin
                              : 0ul;
  const ulong active =
      enabled && !overflow && (config.checked == 0u || prior == 0u)
                           ? min(remaining, window.count)
                           : 0ul;
  if (gid == 0u) {
    if (config.checked == 0u) {
      atomic_store_explicit(&status[0], overflow ? 1u : 0u,
                            memory_order_relaxed);
    }
  }
  const uint base = gid * 4u;
  dispatch_args[base + 0u] =
      uint((active + ulong(config.dispatch_width) - 1ul) /
           ulong(config.dispatch_width));
  dispatch_args[base + 1u] = 1u;
  dispatch_args[base + 2u] = 1u;
  dispatch_args[base + 3u] = uint(active);
}
)metal";
}

[[nodiscard]] inline std::pair<std::uint64_t, std::uint64_t>
MetalMapCheckHash(const MetalMapEncodeResources &resources) noexcept {
  std::uint64_t hi = resources.plan.op_hash_hi ^ 0x6d61702e63686563ull;
  std::uint64_t lo = resources.plan.op_hash_lo ^ 0x6b2e696e64657800ull;
  const auto mix = [](std::uint64_t &hash, const std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
  };
  for (const MetalMapCheck check : resources.checks) {
    const auto *const ref = resources.bindings.resident_inputs.ref(check.binding);
    mix(hi, check.binding);
    mix(hi, check.limit);
    mix(lo, ref == nullptr ? 0u : ref->offset_bytes);
    mix(lo, ref == nullptr ? 0u : ref->stride_bytes);
  }
  return {hi, lo};
}

[[nodiscard]] inline rund::kernel::LoweringArtifact
MetalMapCheckArtifact(const MetalMapEncodeResources &resources) {
  const auto [hi, lo] = MetalMapCheckHash(resources);
  std::string name = "rund_compute_map_";
  rund::kernel::compute_lowering_detail::AppendHex64Digits(name, hi);
  name += "_";
  rund::kernel::compute_lowering_detail::AppendHex64Digits(name, lo);
  std::string source = R"metal(
#include <metal_stdlib>
using namespace metal;

struct ControlConfig {
  uint has_count;
  uint count_u64;
  uint has_predicate;
  uint predicate_u64;
  uint dispatch_width;
  uint checked;
  ulong capacity;
  ulong predicate_expected;
};

kernel void )metal";
  source += name;
  source += "(\n"
            "    device const uchar *count_source [[buffer(0)]],\n"
            "    device const uchar *predicate_source [[buffer(1)]],\n";
  for (std::size_t index = 0u; index < resources.checks.size(); ++index) {
    source += "    device const uchar *indices";
    source += std::to_string(index);
    source += " [[buffer(" + std::to_string(index + 2u) + ")]],\n";
  }
  const std::size_t config_binding = resources.checks.size() + 2u;
  const std::size_t status_binding = config_binding + 1u;
  source += "    constant ControlConfig &config [[buffer(";
  source += std::to_string(config_binding);
  source += ")]],\n    device uint *status [[buffer(";
  source += std::to_string(status_binding);
  source += R"metal()]],
    uint tid [[thread_index_in_threadgroup]]) {
  ulong logical = config.capacity;
  if (config.has_count != 0u) {
    logical = config.count_u64 != 0u
                  ? *reinterpret_cast<device const ulong *>(count_source)
                  : ulong(*reinterpret_cast<device const uint *>(count_source));
  }
  bool enabled = true;
  if (config.has_predicate != 0u) {
    const ulong observed =
        config.predicate_u64 != 0u
            ? *reinterpret_cast<device const ulong *>(predicate_source)
            : ulong(*reinterpret_cast<device const uint *>(predicate_source));
    enabled = observed == config.predicate_expected;
  }
  uint local_invalid = 0xffffffffu;
  if (enabled && logical <= config.capacity) {
    for (ulong ordinal = ulong(tid); ordinal < logical; ordinal += 256ul) {
)metal";
  for (std::size_t index = 0u; index < resources.checks.size(); ++index) {
    const MetalMapCheck check = resources.checks[index];
    const auto *const ref = resources.bindings.resident_inputs.ref(check.binding);
    const std::uint64_t stride = ref == nullptr ? 0u : ref->stride_bytes;
    source += "      if (ulong(*reinterpret_cast<device const uint *>("
              "indices";
    source += std::to_string(index);
    source += " + ordinal * " + std::to_string(stride) + "ul)) >= ";
    source += std::to_string(check.limit);
    source +=
        "ul) { local_invalid = min(local_invalid, uint(min(ordinal, "
        "0xfffffffeul))); }\n";
  }
  source += R"metal(    }
  }
  threadgroup uint invalids[256];
  invalids[tid] = local_invalid;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128u; stride != 0u; stride >>= 1u) {
    if (tid < stride) {
      invalids[tid] = min(invalids[tid], invalids[tid + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid != 0u) { return; }
  status[1] = uint(min(logical, 0xfffffffful));
  status[0] = logical > config.capacity
                  ? 1u
                  : (invalids[0] == 0xffffffffu ? 0u : 2u);
  if (status[0] == 2u) { status[1] = invalids[0]; }
}
)metal";
  rund::kernel::LoweringArtifact artifact{};
  artifact.key.api = rund::kernel::ComputeApi::Metal;
  artifact.key.scalar = resources.plan.scalar;
  artifact.key.domain = resources.plan.domain;
  artifact.key.fixed_format = resources.plan.fixed_format;
  artifact.key.op_hash_hi = hi;
  artifact.key.op_hash_lo = lo;
  artifact.key.canonical_ir_hash_hi = hi;
  artifact.key.canonical_ir_hash_lo = lo;
  artifact.kind = rund::kernel::LoweringArtifactKind::MetalSource;
  artifact.source_text = std::move(source);
  artifact.ok = true;
  artifact.reason = "ok";
  return artifact;
}

[[nodiscard]] inline std::shared_ptr<void>
MetalMapControlPipeline(MetalAdapter &adapter) {
  constexpr const char *key = "map.control.dispatch.v1";
  std::shared_ptr<void> pipeline = LookupMetalNamedPipeline(adapter, key);
  if (pipeline != nullptr) {
    return pipeline;
  }
  id<MTLDevice> const device = (__bridge id<MTLDevice>)adapter.device.get();
  const std::uint64_t begin = MonotonicNanoseconds();
  const std::shared_ptr<void> library_owner =
      AcquireMetalLibrary(adapter, MetalMapControlSource());
  id<MTLLibrary> const library =
      (__bridge id<MTLLibrary>)library_owner.get();
  if (library == nil ||
      !MakeNamedMetalPipeline(device, library, "rund_map_control_dispatch",
                              pipeline)) {
    return {};
  }
  StoreMetalNamedPipeline(adapter, key, pipeline,
                          MonotonicNanoseconds() - begin);
  return pipeline;
}

[[nodiscard]] inline bool PrepareMetalMapControl(
    const rund::AccelDevice &pick, const BoundControl &bound,
    const std::vector<rund::kernel::ComputeDispatchWindow> &windows,
    MetalMapEncodeResources &resources) {
  if (!bound.active() && resources.checks.empty()) {
    return true;
  }
  if (windows.empty() || windows.size() >
                             std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  resources.control = bound.control;
  if (bound.control.has_count()) {
    if (bound.count == nullptr || bound.count_handle == nullptr) {
      return false;
    }
    resources.control_count =
        LookupMetalResidentBuffer(pick, *bound.count, *bound.count_handle);
    if (!resources.control_count.check.ok) {
      return false;
    }
    resources.control_count.ref = *bound.count;
  }
  if (bound.control.has_predicate()) {
    if (bound.predicate == nullptr || bound.predicate_handle == nullptr) {
      return false;
    }
    resources.control_predicate = LookupMetalResidentBuffer(
        pick, *bound.predicate, *bound.predicate_handle);
    if (!resources.control_predicate.check.ok) {
      return false;
    }
    resources.control_predicate.ref = *bound.predicate;
  }
  std::vector<MetalMapControlWindow> authored;
  authored.reserve(windows.size());
  for (const rund::kernel::ComputeDispatchWindow &window : windows) {
    if (window.tile_count > std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
    authored.push_back(MetalMapControlWindow{window.begin_sequence,
                                             window.tile_count});
  }
  const std::uint64_t window_bytes =
      authored.size() * sizeof(MetalMapControlWindow);
  resources.control_config_offset = (window_bytes + 15u) & ~std::uint64_t{15u};
  if (resources.control_config_offset >
      std::numeric_limits<std::uint64_t>::max() -
          sizeof(MetalMapControlConfig)) {
    return false;
  }
  id<MTLComputePipelineState> const map_pipeline =
      (__bridge id<MTLComputePipelineState>)resources.pipeline.get();
  if (map_pipeline == nil) {
    return false;
  }
  const std::uint64_t dispatch_width = std::max<std::uint64_t>(
      1u, std::min<std::uint64_t>(
              windows.front().tile_count,
              [map_pipeline maxTotalThreadsPerThreadgroup]));
  const MetalMapControlConfig config{
      .has_count = bound.control.has_count() ? 1u : 0u,
      .count_u64 =
          bound.control.count_source == rund::kernel::GraphControlSource::U64
              ? 1u
              : 0u,
      .has_predicate = bound.control.has_predicate() ? 1u : 0u,
      .predicate_u64 = bound.control.predicate_source ==
                               rund::kernel::GraphControlSource::U64
                           ? 1u
                           : 0u,
      .dispatch_width = static_cast<std::uint32_t>(dispatch_width),
      .checked = resources.checks.empty() ? 0u : 1u,
      .capacity = bound.control.capacity == 0u
                      ? windows.back().begin_sequence +
                            windows.back().tile_count
                      : bound.control.capacity,
      .predicate_expected = bound.control.predicate_expected,
  };
  resources.control_args = AcquireMetalBuffer(
      *resources.adapter, windows.size() * 4u * sizeof(std::uint32_t),
      MetalBufferUsage::Output);
  resources.control_params = AcquireMetalBuffer(
      *resources.adapter,
      resources.control_config_offset + sizeof(MetalMapControlConfig),
      MetalBufferUsage::Param);
  resources.control_status = AcquireMetalBuffer(
      *resources.adapter, 2u * sizeof(std::uint32_t),
      MetalBufferUsage::Output);
  resources.control_pipeline = MetalMapControlPipeline(*resources.adapter);
  if (!resources.checks.empty()) {
    for (const MetalMapCheck check : resources.checks) {
      const auto *const ref =
          resources.bindings.resident_inputs.ref(check.binding);
      const MetalResidentBufferResult &resident =
          resources.resident.input(check.binding);
      if (check.limit == 0u || ref == nullptr || ref->element_bytes != 4u ||
          ref->stride_bytes < 4u || (ref->stride_bytes & 3u) != 0u ||
          !resident.check.ok || resident.device_buffer == nullptr) {
        return false;
      }
    }
    resources.check_pipeline = MetalPipelineForArtifact(
        *resources.adapter, MetalMapCheckArtifact(resources));
  }
  if (resources.control_args.buffer == nullptr ||
      resources.control_params.buffer == nullptr ||
      resources.control_status.buffer == nullptr ||
      resources.control_pipeline == nullptr ||
      (!resources.checks.empty() && resources.check_pipeline == nullptr) ||
      !UploadMetalBufferUncounted(resources.control_params, authored.data(),
                                  window_bytes)) {
    return false;
  }
  void *const status = MetalBufferContents(resources.control_status);
  if (status == nullptr) {
    return false;
  }
  std::memset(status, 0, 2u * sizeof(std::uint32_t));
  id<MTLBuffer> const params_buffer =
      (__bridge id<MTLBuffer>)resources.control_params.buffer.get();
  auto *const params = static_cast<std::byte *>([params_buffer contents]);
  if (params == nullptr) {
    return false;
  }
  std::memcpy(params + resources.control_config_offset, &config,
              sizeof(config));
  return true;
}

#endif

} // namespace rund::node::accel::detail
