#include <accel/api.hpp>
#include <accel/check.hpp>
#include <accel/device.hpp>

#include "object.hpp"
#include "state.hpp"
#include <limits>
#include <memory>
#include <new>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

namespace {

[[nodiscard]] rund::AccelDevice RejectMetal(const char *const reason) {
  return rund::AccelDevice{
      .check = rund::AccelCheck{false, reason},
      .api = rund::AccelApi::Metal,
      .caps =
          rund::kernel::ComputeCaps{
              .api = rund::kernel::ComputeApi::Metal,
              .reason = reason,
          },
  };
}

} // namespace

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
rund::AccelDevice PickMetal() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      return RejectMetal("accel_metal_device_unavailable");
    }
    NSString *const native_name = device.name;
    const char *const utf8_name =
        native_name == nil ? nullptr : native_name.UTF8String;
    if (utf8_name == nullptr || utf8_name[0] == '\0') {
      return RejectMetal("accel_metal_device_name_invalid");
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      return RejectMetal("accel_metal_queue_unavailable");
    }

    try {
      std::shared_ptr<MetalAdapter> adapter = std::make_shared<MetalAdapter>();
      adapter->device = RetainMetalObject((__bridge void *)device);
      adapter->queue = RetainMetalObject((__bridge void *)queue);
      const std::uint64_t working_set =
          static_cast<std::uint64_t>(device.recommendedMaxWorkingSetSize);
      const std::uint64_t buffer_limit =
          static_cast<std::uint64_t>(device.maxBufferLength);
      adapter->caps = rund::kernel::ComputeCaps{
          .api = rund::kernel::ComputeApi::Metal,
          .device_bytes = working_set != 0u ? working_set : buffer_limit,
          .staging_bytes = 1024u * 1024u,
          .max_window_tiles = std::numeric_limits<rund::kernel::u32>::max(),
          .storage_alignment = sizeof(std::uint32_t),
          .subgroup_width = 1u,
          .ok = true,
          .reason = "ok",
      };
      adapter->info = rund::AccelBackendInfo{
          .device_name = utf8_name,
          .driver_name = "Metal",
          .storage_alignment = adapter->caps.storage_alignment,
          .storage_bytes = buffer_limit,
      };
      adapter->stats = MetalRuntimeStats{.ok = true, .reason = "ok"};

      std::shared_ptr<void> owner = adapter;
      adapter->owner_token = owner;
      return rund::AccelDevice{
          .check = rund::AccelCheck{true, "ok"},
          .api = rund::AccelApi::Metal,
          .caps = adapter->caps,
          .backend =
              rund::kernel::ComputeBackendDispatch{
                  .context = adapter.get(),
                  .execute = ExecuteMetal,
                  .last_error = MetalLastError,
              },
          .backend_info = adapter->info,
          .owner = owner,
      };
    } catch (const std::bad_alloc &) {
      return RejectMetal("accel_metal_device_info_capacity");
    }
  }
}
#else
rund::AccelDevice PickMetal() {
  const char *const reason =
#if defined(__APPLE__)
      "accel_metal_sdk_unavailable";
#else
      "accel_metal_unavailable";
#endif
  return RejectMetal(reason);
}
#endif

} // namespace rund::node::accel::detail
