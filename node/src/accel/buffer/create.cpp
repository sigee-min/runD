#include "../backend/resource.hpp"
#include "../backend/usage.hpp"

#include <accel/buffer.hpp>
#include <accel/check.hpp>
#include <accel/device.hpp>

#include <node/accel/buffer.hpp>

#include <memory>

namespace rund::node::accel {

rund::Buffer CreateBuffer(const rund::AccelDevice &pick,
                          const rund::BufferDesc &desc) {
  if (desc.bytes == 0u) {
    return rund::Buffer{.check =
                            rund::AccelCheck{false, "accel_buffer_bytes_zero"}};
  }
  if (!detail::KnownUsage(desc.usage)) {
    return rund::Buffer{
        .check = rund::AccelCheck{false, "accel_buffer_usage_invalid"}};
  }
  const std::shared_ptr<detail::PickToken> token = detail::AdmitPick(pick);
  return token != nullptr
             ? detail::CreateBackendBuffer(token, desc)
             : rund::Buffer{.check = rund::AccelCheck{
                                false, "accel_buffer_backend_unavailable"}};
}

} // namespace rund::node::accel
