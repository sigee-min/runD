#include "transfer.hpp"
#include "../backend/resource.hpp"
#include "../kernel/prepared/interface/api.hpp"
#include "local.hpp"

#include <cstdint>

namespace rund::node::accel {

rund::AccelCheck UploadAccelBuffer(const rund::AccelContext &context,
                                   const rund::AccelBuffer &buffer,
                                   const void *const data,
                                   const std::uint64_t bytes,
                                   const std::uint64_t offset) {
  const detail::TransferAdmission admission =
      detail::AdmitAccelBufferTransfer(context, buffer);
  if (!admission.check.ok) {
    return admission.check;
  }
  if (!rund::kernel::checked::add(offset, bytes) ||
      offset + bytes > admission.byte_extent) {
    return detail::RejectAccelCheck("accel_buffer_upload_overflow");
  }
  return detail::TransferCheckFrom(
      detail::UploadBackendBuffer(admission.pick, admission.route.ref,
                                  admission.route.handle, data, bytes, offset),
      "accel_buffer_upload_overflow");
}

rund::AccelCheck DownloadAccelBuffer(const rund::AccelContext &context,
                                     const rund::AccelBuffer &buffer,
                                     void *const data,
                                     const std::uint64_t bytes,
                                     const std::uint64_t offset) {
  return detail::DownloadAccelBufferMeasured(context, buffer, data, bytes,
                                             offset)
      .check;
}

} // namespace rund::node::accel
