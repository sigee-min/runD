#pragma once

#include <accel/api.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/graph/visibility.hpp>
#include <accel/kernel/value.hpp>

#include "../../context/shared.hpp"
#include "../token.hpp"
#include <memory>
#include <span>
#include <vector>

namespace rund::node::accel::detail {

struct KernelToken {
  std::uint64_t kernel_id = 0u;
  std::uint64_t context_id = 0u;
  std::uint64_t graph_id_hi = 0u;
  std::uint64_t graph_id_lo = 0u;
  std::uint64_t node_count = 0u;
  rund::AccelApi api = rund::AccelApi::Auto;
  rund::kernel::ComputeScalar scalar = rund::kernel::ComputeScalar::Lane32;
  rund::kernel::ComputeDomain domain = rund::kernel::ComputeDomain::Fixed;
  rund::kernel::ComputeCaps frozen_caps{};
  std::shared_ptr<void> context_owner{};
  std::vector<rund::kernel::BufferRole> graph_roles{};
  std::vector<rund::AccelBufferDesc> graph_shapes{};
  std::vector<rund::GraphBufferVisibility> graph_visibilities{};
  std::vector<std::uint64_t> graph_alias_representatives{};
  std::vector<ResetPlan> resets{};
  std::vector<KernelExecutionStep> steps{};
  std::vector<std::uint8_t> required_barriers{};
  std::uint64_t removed_dispatch_count = 0u;
  std::uint64_t original_operation_count = 0u;
  std::uint64_t fused_operation_count = 0u;
  std::uint64_t fusion_rejection_count = 0u;
  const char *fusion_reason = "compute_fusion_invalid";
};

struct KernelTokenAdmission {
  KernelAdmission admission{};
  std::shared_ptr<KernelToken> token{};
};

[[nodiscard]] std::shared_ptr<KernelToken>
MakeKernelToken(KernelToken token) noexcept;
[[nodiscard]] std::shared_ptr<KernelToken>
LookupKernelToken(const std::shared_ptr<void> &owner,
                  std::uint64_t kernel_id) noexcept;
[[nodiscard]] KernelAdmission RejectAdmission(const char *reason) noexcept;
[[nodiscard]] KernelTokenAdmission
AdmitKernelTokenWithContext(const rund::AccelKernel &kernel,
                            const ContextAdmission &context_admission);
[[nodiscard]] KernelAdmission
AdmitKernelWithContext(const rund::AccelKernel &kernel,
                       const ContextAdmission &context_admission);

} // namespace rund::node::accel::detail
