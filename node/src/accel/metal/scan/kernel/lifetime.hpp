#pragma once

#include "local.hpp"

#include <utility>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
void DestroyMetalScanEncodeResources(void* const raw) {
  auto* const resources = static_cast<MetalScanEncodeResources*>(raw);
  if (resources == nullptr) { return; }
  if (resources->adapter != nullptr) {
    MetalAdapter& adapter = *resources->adapter;
    ReleaseMetalBuffer(adapter, std::move(resources->totals));
    ReleaseMetalBuffer(adapter, std::move(resources->status));
  }
  delete resources;
}
#endif

}  // namespace rund::node::accel::detail
