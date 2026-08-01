#include <accel/device.hpp>

#include "../../../backend/result.hpp"
#include "../../command/submit.hpp"
#include "../../number.hpp"
#include "../../resident/access.hpp"
#include "find.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <utility>
#include <vector>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
BackendCopy
CopyMetalResidentBuffers(const rund::AccelDevice &pick,
                         const std::span<const CopyRoute> requests) {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr || requests.empty()) {
    return {};
  }
  struct CopyPlan final {
    std::shared_ptr<void> source_owner;
    std::shared_ptr<void> target_owner;
    id<MTLBuffer> source = nil;
    id<MTLBuffer> target = nil;
    NSUInteger source_offset = 0u;
    NSUInteger target_offset = 0u;
    NSUInteger bytes = 0u;
  };
  @autoreleasepool {
    try {
      std::vector<CopyPlan> plans;
      plans.reserve(requests.size());
      {
        MetalResidentState &resident = MetalResidents(*adapter);
        std::lock_guard resident_lock{resident.mutex};
        for (const CopyRoute &request : requests) {
          MetalResidentBufferResult source = ResolveMetalResidentBuffer(
              resident, request.source, request.source_handle,
              "accel_buffer_unavailable");
          MetalResidentBufferResult target = ResolveMetalResidentBuffer(
              resident, request.target, request.target_handle,
              "accel_buffer_unavailable");
          if (!source.check.ok || !target.check.ok ||
              source.device_buffer == nullptr ||
              target.device_buffer == nullptr ||
              request.source_offset > request.source.bytes ||
              request.bytes > request.source.bytes - request.source_offset ||
              request.target_offset > request.target.bytes ||
              request.bytes > request.target.bytes - request.target_offset ||
              request.source.offset_bytes >
                  std::numeric_limits<std::uint64_t>::max() -
                      request.source_offset ||
              request.target.offset_bytes >
                  std::numeric_limits<std::uint64_t>::max() -
                      request.target_offset) {
            const char *const reason = !source.check.ok ? source.check.reason
                                       : !target.check.ok
                                           ? target.check.reason
                                           : "accel_buffer_copy_overflow";
            return BackendCopy{.check = {false, reason}};
          }
          if (request.bytes == 0u) {
            continue;
          }
          const std::uint64_t source_offset =
              request.source.offset_bytes + request.source_offset;
          const std::uint64_t target_offset =
              request.target.offset_bytes + request.target_offset;
          CopyPlan plan{
              .source_owner = std::move(source.device_buffer),
              .target_owner = std::move(target.device_buffer),
          };
          if (!ToNSUInteger(source_offset, plan.source_offset) ||
              !ToNSUInteger(target_offset, plan.target_offset) ||
              !ToNSUInteger(request.bytes, plan.bytes)) {
            return BackendCopy{.check = {false, "accel_buffer_copy_overflow"}};
          }
          plan.source = (__bridge id<MTLBuffer>)plan.source_owner.get();
          plan.target = (__bridge id<MTLBuffer>)plan.target_owner.get();
          if (plan.source == nil || plan.target == nil) {
            return BackendCopy{.check = {false, "accel_buffer_unavailable"}};
          }
          plans.push_back(std::move(plan));
        }
      }
      const auto overlaps = [](const NSUInteger left_offset,
                               const NSUInteger left_bytes,
                               const NSUInteger right_offset,
                               const NSUInteger right_bytes) noexcept {
        return left_offset < right_offset + right_bytes &&
               right_offset < left_offset + left_bytes;
      };
      for (std::size_t target_index = 0u; target_index < plans.size();
           ++target_index) {
        const CopyPlan &target = plans[target_index];
        for (std::size_t source_index = 0u; source_index < plans.size();
             ++source_index) {
          const CopyPlan &source = plans[source_index];
          if (target.target == source.source &&
              overlaps(target.target_offset, target.bytes, source.source_offset,
                       source.bytes)) {
            return BackendCopy{.check = {false, "accel_buffer_copy_overlap"}};
          }
          if (source_index != target_index && target.target == source.target &&
              overlaps(target.target_offset, target.bytes, source.target_offset,
                       source.bytes)) {
            return BackendCopy{.check = {false, "accel_buffer_copy_overlap"}};
          }
        }
      }
      if (plans.empty()) {
        return BackendCopy{.check = {true, "ok"}};
      }
      id<MTLCommandQueue> queue =
          (__bridge id<MTLCommandQueue>)adapter->queue.get();
      id<MTLCommandBuffer> command = queue == nil ? nil : [queue commandBuffer];
      id<MTLBlitCommandEncoder> encoder =
          command == nil ? nil : [command blitCommandEncoder];
      if (command == nil || encoder == nil) {
        return BackendCopy{.check = {false, "accel_metal_command_unavailable"}};
      }
      for (const CopyPlan &plan : plans) {
        [encoder copyFromBuffer:plan.source
                   sourceOffset:plan.source_offset
                       toBuffer:plan.target
              destinationOffset:plan.target_offset
                           size:plan.bytes];
      }
      [encoder endEncoding];
      const rund::AccelCheck copied =
          WaitCommand(*adapter, (__bridge void *)command);
      return BackendCopy{.check = copied, .command_submits = 1u};
    } catch (const std::bad_alloc &) {
      return BackendCopy{.check = {false, "accel_buffer_unavailable"}};
    }
  }
}
#else
BackendCopy CopyMetalResidentBuffers(const rund::AccelDevice &,
                                     const std::span<const CopyRoute>) {
  return {};
}
#endif

} // namespace rund::node::accel::detail
