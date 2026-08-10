#include <accel/api.hpp>
#include <accel/device.hpp>

#include "../resident/access.hpp"
#include "../resident/storage.hpp"
#include "owner.hpp"

#include <mutex>
#include <new>
#include <utility>

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

void MetalResidentOwnerDelete::operator()(
    MetalResidentOwner *const value) const noexcept {
  delete value;
}

std::shared_ptr<MetalResidentOwner> MakeMetalResidentOwner() noexcept {
  MetalResidentOwner *const raw = new (std::nothrow) MetalResidentOwner{};
  if (raw == nullptr) {
    return {};
  }
  try {
    std::unique_ptr<MetalResidentOwner, MetalResidentOwnerDelete> owner{
        raw, MetalResidentOwnerDelete{.owner = raw}};
    return std::shared_ptr<MetalResidentOwner>{std::move(owner)};
  } catch (...) {
    return {};
  }
}

std::shared_ptr<MetalResidentOwner>
LookupMetalResidentOwner(const std::shared_ptr<void> &owner) noexcept {
  if (owner == nullptr) {
    return {};
  }
  const auto *const capability =
      std::get_deleter<MetalResidentOwnerDelete>(owner);
  return capability == nullptr || capability->owner == nullptr
             ? std::shared_ptr<MetalResidentOwner>{}
             : std::shared_ptr<MetalResidentOwner>{owner, capability->owner};
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
