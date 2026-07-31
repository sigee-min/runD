#pragma once

#include <accel/context/value.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/runtime.hpp>

#include <node/accel/context.hpp>

#include "../context/internal.hpp"
#include <cstdint>

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelEvidence
RejectKernelEvidence(const rund::AccelContext &context,
                     const KernelExecution &execution, const char *reason);

[[nodiscard]] rund::AccelEvidence EvidenceFromStats(
    const rund::AccelContext &context, const KernelExecution &execution,
    const rund::RuntimeStats &stats, std::uint64_t original_dispatch_count,
    std::uint64_t final_dispatch_count, bool ok, const char *reason,
    std::uint64_t internal_producer_consumer_roundtrip_bytes = 0u,
    std::uint64_t external_producer_consumer_roundtrip_bytes = 0u,
    std::uint64_t failed_batches = 0u, std::uint64_t first_failed_batch = 0u,
    std::uint32_t first_status = 0u);

} // namespace rund::node::accel::detail
