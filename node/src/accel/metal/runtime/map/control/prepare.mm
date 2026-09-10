#include "../../api.hpp"
#include "../control.hpp"
#include "../resources.hpp"

#include "../../../../kernel/backend/run.hpp"
#include "../../../buffer/pool/acquire.hpp"
#include "../../../resident.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

bool PrepareMetalMapControl(
    const rund::AccelDevice &pick, const BoundControl &bound,
    const std::vector<rund::kernel::ComputeDispatchWindow> &windows,
    MetalMapEncodeResources &resources) {
  if (!bound.active() && resources.prepared->checks.empty()) {
    return true;
  }
  if (windows.empty() ||
      windows.size() > std::numeric_limits<std::uint32_t>::max()) {
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
    authored.push_back(
        MetalMapControlWindow{window.begin_sequence, window.tile_count});
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
      resources.prepared == nullptr ? nil
                                    : (__bridge id<MTLComputePipelineState>)
                                          resources.prepared->pipeline.get();
  if (map_pipeline == nil) {
    return false;
  }
  const std::uint64_t dispatch_width = std::max<std::uint64_t>(
      1u,
      std::min<std::uint64_t>(windows.front().tile_count,
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
      .checked = resources.prepared->checks.empty() ? 0u : 1u,
      .capacity = bound.control.capacity == 0u ? windows.back().begin_sequence +
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
      *resources.adapter, 2u * sizeof(std::uint32_t), MetalBufferUsage::Output);
  resources.control_pipeline = resources.prepared->control_pipeline;
  if (!resources.prepared->checks.empty()) {
    for (const MetalMapCheck check : resources.prepared->checks) {
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
    resources.check_pipeline = resources.prepared->check_pipeline;
  }
  if (resources.control_args.buffer == nullptr ||
      resources.control_params.buffer == nullptr ||
      resources.control_status.buffer == nullptr ||
      resources.control_pipeline == nullptr ||
      (!resources.prepared->checks.empty() &&
       resources.check_pipeline == nullptr) ||
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
