#pragma once

#include "../local.hpp"

#include <utility>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
void DestroyMetalMapEncodeResources(void* const raw) {
  auto* const resources = static_cast<MetalMapEncodeResources*>(raw);
  if (resources == nullptr) { return; }
  if (resources->adapter != nullptr) {
    ReleaseMetalBuffer(*resources->adapter, std::move(resources->param));
    ReleaseMetalBuffer(*resources->adapter,
                       std::move(resources->control_args));
    ReleaseMetalBuffer(*resources->adapter,
                       std::move(resources->control_params));
    ReleaseMetalBuffer(*resources->adapter,
                       std::move(resources->control_status));
  }
  delete resources;
}
#endif

}  // namespace rund::node::accel::detail
