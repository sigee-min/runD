#pragma once

#include <accel/device.hpp>

#include "../../resident/model.hpp"

#include <cstddef>
#include <memory>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
struct MetalResidentReq {
  const rund::kernel::ResidentBufferRef *ref = nullptr;
  const std::shared_ptr<void> *handle = nullptr;
  MetalResidentBufferResult *out = nullptr;
};

void LookupMetalResidentBatch(const rund::AccelDevice &pick,
                              MetalResidentReq *reqs, std::size_t count,
                              const char *missing_reason);

template <std::size_t N>
inline void LookupMetalResidentBatch(const rund::AccelDevice &pick,
                                     MetalResidentReq (&reqs)[N],
                                     const char *const missing_reason) {
  LookupMetalResidentBatch(pick, reqs, N, missing_reason);
}
#endif

} // namespace rund::node::accel::detail
