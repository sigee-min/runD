#include "object.hpp"
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Foundation/Foundation.h>
#endif

namespace rund::node::accel::detail {

std::shared_ptr<void> RetainMetalObject(void* const object) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  if (object == nullptr) {
    return {};
  }
  id metal_object = (__bridge id)object;
  if (metal_object == nil) {
    return {};
  }
  void* const retained = const_cast<void*>(CFBridgingRetain(metal_object));
  return std::shared_ptr<void>(retained, [](void* handle) {
    if (handle != nullptr) {
      CFBridgingRelease(handle);
    }
  });
#else
  (void)object;
  return {};
#endif
}

}  // namespace rund::node::accel::detail
