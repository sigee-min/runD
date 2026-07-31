#include "buffer.hpp"
#include <rund/counter.hpp>
#include "../backend/match.hpp"
#include "../backend/usage.hpp"
#include "../resident/ref.hpp"
#include "../resident/slot.hpp"
#include "buffer/find.hpp"

namespace rund::node::accel::detail {

bool CpuPickOwnsAdapter(const rund::AccelDevice &pick) noexcept {
  if (!pick.check.ok || pick.api != rund::AccelApi::Cpu ||
      pick.owner == nullptr || pick.backend.context == nullptr ||
      pick.backend.execute != ExecuteCpu ||
      pick.owner.get() != pick.backend.context) {
    return false;
  }
  const auto *const adapter =
      static_cast<const CpuAdapter *>(pick.backend.context);
  return SameOwner(adapter->owner_token, pick.owner);
}

CpuAdapter *CpuAdapterFromPick(const rund::AccelDevice &pick) noexcept {
  return CpuPickOwnsAdapter(pick)
             ? static_cast<CpuAdapter *>(pick.backend.context)
             : nullptr;
}

CpuBufferResult CreateCpuResidentBuffer(const rund::AccelDevice &pick,
                                        const rund::BufferDesc &desc) {
  CpuAdapter *const adapter = CpuAdapterFromPick(pick);
  if (adapter == nullptr) {
    return RejectResident<CpuBufferResult>("accel_buffer_backend_unavailable");
  }
  auto buffer = std::make_shared<CpuBuffer>();
  {
    std::lock_guard<std::mutex> lock{adapter->mutex};
    if (!ResidentAppendId(adapter->buffers, adapter->next_resident_id)) {
      return RejectResident<CpuBufferResult>(
          "accel_buffer_backend_unavailable");
    }
    buffer->id = adapter->next_resident_id++;
    buffer->bytes = desc.bytes;
    buffer->element_bytes = 1u;
    buffer->stride_bytes = 1u;
    buffer->count = desc.bytes;
    buffer->usage = ResidentUsage(desc.usage);
    buffer->read_capable = desc.usage != rund::BufferUsage::WriteOnly;
    buffer->write_capable = desc.usage != rund::BufferUsage::ReadOnly;
    buffer->owner = buffer;
    adapter->buffers.push_back(buffer);
    ::rund::detail::counter::Accumulate(adapter->buffer_allocation_count, 1u);
  }
  buffer->data.resize(static_cast<std::size_t>(desc.bytes));
  return CpuBufferResult{
      .check = rund::AccelCheck{true, "ok"},
      .ref = RefFromResident(*buffer),
      .buffer = std::move(buffer),
  };
}

} // namespace rund::node::accel::detail
