#include <accel/api.hpp>
#include <accel/device.hpp>

#include "../resident/access.hpp"
#include "../resident/storage.hpp"
#include "owner.hpp"

#include <mutex>

namespace rund::node::accel::detail {

MetalResidentOwner::~MetalResidentOwner() {
  if (adapter == nullptr) {
    return;
  }
  MetalResidentState &resident = MetalResidents(*adapter);
  std::lock_guard lock{resident.mutex};
  resident.buffers.erase(id);
  adapter = nullptr;
  id = 0u;
}

bool MetalPickOwnsAdapter(const rund::AccelDevice &pick) noexcept {
  if (!pick.check.ok || pick.api != rund::AccelApi::Metal ||
      pick.owner == nullptr || pick.backend.context == nullptr ||
      pick.backend.execute != ExecuteMetal ||
      pick.owner.get() != pick.backend.context) {
    return false;
  }
  const auto *const adapter =
      static_cast<const MetalAdapter *>(pick.backend.context);
  return SameOwner(adapter->owner_token, pick.owner);
}

} // namespace rund::node::accel::detail
