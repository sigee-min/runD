#include "internal.hpp"

#include "../../buffer/local.hpp"
#include "../../device/state.hpp"
#include "../../type.hpp"

#include <kernel/core/checked.hpp>

#include <limits>
#include <memory>
#include <new>

namespace rund::compute::detail {

Result<std::shared_ptr<VirtualBacking>>
make_resident_virtual_backing(const std::shared_ptr<DeviceState> &device,
                              const Type type,
                              const std::uint64_t count) noexcept {
  if (device == nullptr || !valid_type(type) ||
      count > std::numeric_limits<std::size_t>::max()) {
    return Result<std::shared_ptr<VirtualBacking>>::fail(
        Reason::BufferCapacity);
  }
  const std::uint64_t element_bytes = type_bytes(type);
  std::uint64_t logical_bytes = 0u;
  if (element_bytes == 0u ||
      !kernel::checked::mul(count, element_bytes, logical_bytes)) {
    return Result<std::shared_ptr<VirtualBacking>>::fail(
        Reason::BufferCapacity);
  }
  const auto storage = planned_buffer_storage_bytes(*device, logical_bytes);
  if (!storage) {
    return Result<std::shared_ptr<VirtualBacking>>::fail(storage.reason());
  }
  auto resident = make_planned_residency_buffer(
      device, type, static_cast<std::size_t>(count), *storage);
  if (!resident) {
    return Result<std::shared_ptr<VirtualBacking>>::fail(resident.reason());
  }
  try {
    auto backing = std::make_shared<ResidentVirtualBacking>();
    VirtualBackingAccess::bind_resident(*backing, std::move(resident).value());
    return Result<std::shared_ptr<VirtualBacking>>::success(std::move(backing));
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<VirtualBacking>>::fail(
        Reason::BufferCapacity);
  }
}

} // namespace rund::compute::detail
