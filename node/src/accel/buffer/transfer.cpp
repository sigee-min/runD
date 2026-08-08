#include <accel/buffer.hpp>
#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../backend/resource.hpp"

#include <node/accel/buffer.hpp>

#include <memory>

namespace rund::node::accel {

rund::AccelCheck UploadBuffer(const rund::AccelDevice &pick,
                              const rund::Buffer &buffer, const void *data,
                              const std::uint64_t bytes,
                              const std::uint64_t offset) {
  const std::shared_ptr<detail::PickToken> token = detail::AdmitPick(pick);
  return token != nullptr
             ? detail::UploadBackendBuffer(token, buffer, data, bytes, offset)
             : rund::AccelCheck{false, "accel_buffer_backend_unavailable"};
}

rund::AccelCheck DownloadBuffer(const rund::AccelDevice &pick,
                                const rund::Buffer &buffer, void *data,
                                const std::uint64_t bytes,
                                const std::uint64_t offset) {
  const std::shared_ptr<detail::PickToken> token = detail::AdmitPick(pick);
  return token != nullptr
             ? detail::DownloadBackendBuffer(token, buffer, data, bytes, offset)
                   .check
             : rund::AccelCheck{false, "accel_buffer_backend_unavailable"};
}

} // namespace rund::node::accel
