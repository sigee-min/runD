#pragma once

#include <rund/compute.hpp>

#include <cstdint>
#include <span>

namespace rund_node_test_compute_reuse {

using ReuseProgram = rund::compute::Program<std::int32_t(std::int32_t)>;

[[nodiscard]] bool SameMemory(const rund::compute::MemoryStats &,
                              const rund::compute::MemoryStats &) noexcept;
[[nodiscard]] bool CheckTelemetryMath();
[[nodiscard]] bool CheckHostIdentities(rund::compute::Device &);
[[nodiscard]] bool CheckInitialHostReuse(ReuseProgram &,
                                         std::span<const std::int32_t>);
[[nodiscard]] bool CheckProgramConcurrency(rund::compute::Device &);
[[nodiscard]] bool CheckProgramLifetime(rund::compute::Device &);
[[nodiscard]] int CheckReadOnlyOneShot(rund::compute::Device &);
[[nodiscard]] int CheckBufferReuseAndReceipt(rund::compute::Device &,
                                             ReuseProgram &,
                                             std::span<const std::int32_t>);

} // namespace rund_node_test_compute_reuse
