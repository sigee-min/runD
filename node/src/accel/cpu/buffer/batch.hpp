#pragma once

#include <accel/device.hpp>

#include "../buffer.hpp"

#include <cstddef>
#include <memory>

namespace rund::node::accel::detail {

struct CpuResidentReq {
  const rund::kernel::ResidentBufferRef *ref = nullptr;
  const std::shared_ptr<void> *handle = nullptr;
  std::uint32_t usage = 0u;
  CpuBufferResult *out = nullptr;
};

void LookupCpuResidentBatch(const rund::AccelDevice &pick, CpuResidentReq *reqs,
                            std::size_t count);

template <std::size_t N>
inline void LookupCpuResidentBatch(const rund::AccelDevice &pick,
                                   CpuResidentReq (&reqs)[N]) {
  LookupCpuResidentBatch(pick, reqs, N);
}

} // namespace rund::node::accel::detail
