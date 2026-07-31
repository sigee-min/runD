#include <accel/buffer.hpp>
#include <accel/check.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/context/evidence.hpp>
#include <accel/context/value.hpp>

#include "local.hpp"

namespace rund::node::accel::detail {

rund::AccelContext RejectContext(const char *const reason) {
  return rund::AccelContext{
      .check = rund::AccelCheck{false, reason},
      .evidence = rund::AccelContextEvidence{.ok = false, .reason = reason},
  };
}

rund::AccelBuffer RejectBuffer(const rund::AccelBufferDesc &desc,
                               const rund::Buffer &buffer,
                               const char *const reason) {
  return rund::AccelBuffer{
      .check = rund::AccelCheck{false, reason},
      .buffer = buffer,
      .scalar_width_bytes = desc.scalar_width_bytes,
      .count = desc.count,
      .usage = desc.usage,
      .owner = buffer.owner,
      .handle = buffer.handle,
      .reason = reason,
  };
}

} // namespace rund::node::accel::detail
