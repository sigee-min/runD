#pragma once

#include "state.hpp"

#include "../../numeric/policy.hpp"
#include "../numeric.hpp"

#include <accel/check.hpp>
#include <accel/device.hpp>

#include <kernel/program/compute/transform/twiddle.hpp>

#include <cstdint>
#include <span>
#include <string>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
template <typename Hash>
[[nodiscard]] rund::kernel::ComputePlan
NumericPseudoPlan(const Hash hash,
                  const rund::kernel::ComputeScalar scalar,
                  const rund::kernel::ComputeDomain domain,
                  const rund::kernel::ComputeFixedFormat fixed_format = {})
    noexcept {
  return rund::kernel::ComputePlan{
      .op_hash_hi = hash.hi,
      .op_hash_lo = hash.lo,
      .api = rund::kernel::ComputeApi::Vulkan,
      .scalar = scalar,
      .domain = domain,
      .fixed_format = fixed_format,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] rund::AccelCheck RejectElement(VulkanAdapter &adapter,
                                             const char *reason);
[[nodiscard]] VulkanCollectivePipeline *AcquireNumericPipeline(
    VulkanAdapter &adapter, std::uint32_t descriptor_count,
    std::uint32_t push_bytes, const rund::kernel::ComputePlan &pseudo,
    std::string source, NumericPolicy policy);
[[nodiscard]] rund::AccelCheck StatusCheck(VulkanAdapter &adapter,
                                           const VulkanBuffer &status,
                                           rund::kernel::u64 count);
[[nodiscard]] VulkanStorageBinding
ResidentBinding(const VulkanResidentBufferResult &resident) noexcept;
void DestroyNumericPrepared(void *raw);
[[nodiscard]] bool PrepareTwiddle(VulkanNumericPrepared &state,
                                  const rund::kernel::TransformPlan &plan);
[[nodiscard]] rund::AccelCheck
LookupPrepared(const rund::AccelDevice &pick, VulkanNumericPrepared &state,
               std::span<VulkanResidentReq> requests);
[[nodiscard]] rund::AccelCheck
FinalizePrepared(VulkanNumericPrepared &state, const NumericParams &params,
                 std::uint32_t descriptor_count,
                 VulkanCollectivePipeline *pipeline,
                 KernelPreparationMode mode);
#endif

} // namespace rund::node::accel::detail
