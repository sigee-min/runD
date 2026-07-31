#include <accel/device.hpp>

#include "../resident/result.hpp"
#include "buffer.hpp"
#include "buffer/batch.hpp"

namespace rund::node::accel::detail {

CpuBufferResult
LookupCpuResidentView(const rund::AccelDevice &pick,
                      const rund::kernel::ResidentBufferRef &requested,
                      const std::shared_ptr<void> &handle,
                      const std::uint32_t usage) {
  CpuBufferResult result{};
  CpuResidentReq reqs[] = {
      CpuResidentReq{
          .ref = &requested, .handle = &handle, .usage = usage, .out = &result},
  };
  LookupCpuResidentBatch(pick, reqs);
  return result;
}

CpuBufferResult
LookupCpuResidentBuffer(const rund::AccelDevice &pick,
                        const rund::kernel::ResidentBufferRef &requested,
                        const std::shared_ptr<void> &handle) {
  CpuBufferResult result =
      LookupCpuResidentView(pick, requested, handle, requested.usage);
  if (!result.check.ok) {
    return result;
  }
  const CpuBuffer &buffer = *result.buffer;
  if (buffer.element_bytes != requested.element_bytes ||
      buffer.stride_bytes != requested.stride_bytes ||
      buffer.count != requested.count || buffer.usage != requested.usage) {
    return RejectResident<CpuBufferResult>("accel_context_buffer_invalid");
  }
  return result;
}

} // namespace rund::node::accel::detail
