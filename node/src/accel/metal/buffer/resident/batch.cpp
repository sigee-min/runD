#include "batch.hpp"

#include "../../resident/access.hpp"
#include "find.hpp"

#include <mutex>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
void LookupMetalResidentBatch(const rund::AccelDevice &pick,
                              MetalResidentReq *const reqs,
                              const std::size_t count,
                              const char *const missing_reason) {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  for (std::size_t index = 0u; index < count; ++index) {
    MetalResidentReq &req = reqs[index];
    if (req.out == nullptr) {
      continue;
    }
    const bool handle_valid = req.handle != nullptr && *req.handle != nullptr;
    *req.out = RejectResident<MetalResidentBufferResult>(
        adapter != nullptr && handle_valid
            ? missing_reason
            : "accel_metal_resident_owner_invalid");
  }
  if (adapter == nullptr) {
    return;
  }
  MetalResidentState &resident = MetalResidents(*adapter);
  std::lock_guard lock{resident.mutex};
  for (std::size_t index = 0u; index < count; ++index) {
    MetalResidentReq &req = reqs[index];
    if (req.ref == nullptr || req.handle == nullptr || req.out == nullptr ||
        *req.handle == nullptr) {
      continue;
    }
    *req.out = ResolveMetalResidentBuffer(resident, *req.ref, *req.handle,
                                          missing_reason);
    // Resolution proves the requested window fits the owning allocation, but
    // the resolver's canonical result names the whole allocation.  Native
    // primitive encoders need the requested subview offset/range to bind a
    // contiguous View without materializing a second buffer.
    if (req.out->check.ok) {
      req.out->ref = *req.ref;
    }
  }
}
#endif

} // namespace rund::node::accel::detail
