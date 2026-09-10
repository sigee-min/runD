#include "../internal.hpp"

#include <new>
#include <utility>

namespace rund::node::accel::detail::vulkan_device_vsm {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

void OwnerDelete::operator()(Owner *const value) const noexcept {
  if (value != nullptr && value->adapter != nullptr) {
    DestroyCommand(value->adapter->device, value->graph_secondary);
    ReleaseVulkanLeases(value->descriptor_leases);
  }
  delete value;
}

std::shared_ptr<Owner> make_owner() noexcept {
  Owner *const raw = new (std::nothrow) Owner{};
  if (raw == nullptr) {
    return {};
  }
  try {
    std::unique_ptr<Owner, OwnerDelete> owner{raw, OwnerDelete{.owner = raw}};
    return std::shared_ptr<Owner>{std::move(owner)};
  } catch (...) {
    return {};
  }
}

std::shared_ptr<Owner>
owner_of(const std::shared_ptr<void> &lowering) noexcept {
  if (lowering == nullptr) {
    return {};
  }
  const auto *const capability = std::get_deleter<OwnerDelete>(lowering);
  return capability == nullptr || capability->owner == nullptr
             ? std::shared_ptr<Owner>{}
             : std::shared_ptr<Owner>{lowering, capability->owner};
}

#endif

} // namespace rund::node::accel::detail::vulkan_device_vsm
