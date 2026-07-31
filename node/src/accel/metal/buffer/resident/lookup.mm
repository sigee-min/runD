#include <accel/device.hpp>

#include "../../resident/access.hpp"
#include "find.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
MetalResidentBufferResult
LookupMetalResidentBuffer(const rund::AccelDevice &pick,
                          const rund::kernel::ResidentBufferRef &ref,
                          const std::shared_ptr<void> &handle) {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr || handle == nullptr) {
    return RejectResident<MetalResidentBufferResult>(
        "accel_metal_resident_owner_invalid");
  }
  MetalResidentState &resident = MetalResidents(*adapter);
  std::lock_guard lock{resident.mutex};
  return ResolveMetalResidentBuffer(resident, ref, handle,
                                    "accel_metal_resident_id_unavailable");
}
#else
MetalResidentBufferResult
LookupMetalResidentBuffer(const rund::AccelDevice &,
                          const rund::kernel::ResidentBufferRef &,
                          const std::shared_ptr<void> &) {
  return RejectResident<MetalResidentBufferResult>("accel_metal_unavailable");
}
#endif

} // namespace rund::node::accel::detail
