#include "batch.hpp"

#include "../../resident/slot.hpp"
#include "find.hpp"

#include <mutex>
#include <utility>

namespace rund::node::accel::detail {

void LookupCpuResidentBatch(const rund::AccelDevice &pick,
                            CpuResidentReq *const reqs,
                            const std::size_t count) {
  CpuAdapter *const adapter = CpuAdapterFromPick(pick);
  for (std::size_t index = 0u; index < count; ++index) {
    CpuResidentReq &req = reqs[index];
    if (req.out != nullptr) {
      *req.out =
          RejectResident<CpuBufferResult>("accel_context_buffer_invalid");
    }
  }
  if (adapter == nullptr) {
    return;
  }

  std::lock_guard<std::mutex> lock{adapter->mutex};
  for (std::size_t index = 0u; index < count; ++index) {
    CpuResidentReq &req = reqs[index];
    if (req.ref == nullptr || req.handle == nullptr || req.out == nullptr) {
      continue;
    }
    const auto *const slot = ResidentIndexedSlot(adapter->buffers, req.ref->id);
    if (slot == nullptr) {
      *req.out = RejectResident<CpuBufferResult>("compute_resident_id_invalid");
      continue;
    }
    std::shared_ptr<void> owner = slot->lock();
    if (owner == nullptr) {
      *req.out = RejectResident<CpuBufferResult>("accel_buffer_unavailable");
      continue;
    }
    const auto *const buffer = static_cast<const CpuBuffer *>(owner.get());
    const char *reason = "ok";
    if (req.ref->usage != req.usage) {
      reason = "compute_resident_usage_invalid";
    } else if (ResidentRefFits(*buffer, owner, *req.ref, *req.handle,
                               "compute_resident_id_invalid", reason, true)) {
      if (buffer->data.size() >= buffer->bytes) {
        *req.out = CpuResidentResult(std::move(owner));
        continue;
      }
      reason = "accel_buffer_unavailable";
    }
    *req.out = RejectResident<CpuBufferResult>(reason);
  }
}

} // namespace rund::node::accel::detail
