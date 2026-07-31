#pragma once

#include <accel/device.hpp>

#include <node/accel/cpu/simd.hpp>

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/binding/model.hpp>
#include <kernel/program/compute/cpu.hpp>
#include <kernel/program/compute/ir.hpp>
#include <kernel/program/compute/lowering/admission.hpp>
#include <kernel/program/compute/model.hpp>

#include <memory>
#include <mutex>
#include <vector>

namespace rund::node::accel::detail {

struct CpuProfile {
  bool sse2 = false;
  bool neon = false;
  bool ok = false;
  const char *reason = "accel_cpu_simd_strategy_unavailable";
  const char *source = "unsupported_target";
};

struct CpuAdapter {
  rund::kernel::ComputeCaps generic_caps{};
  rund::kernel::CpuCaps caps{};
  rund::AccelBackendInfo info{};
  std::weak_ptr<void> owner_token{};
  std::mutex mutex{};
  std::vector<std::weak_ptr<void>> buffers{};
  std::shared_ptr<struct CpuSortScratch> sort_scratch{};
  std::shared_ptr<struct CpuScatterScratch> scatter_scratch{};
  std::vector<rund::kernel::u32> scatter_reduce_indices{};
  std::uint64_t next_resident_id = 1u;
  std::uint64_t dispatch_count = 0u;
  std::uint64_t reset_command_count = 0u;
  std::uint64_t reset_bytes = 0u;
  std::uint64_t buffer_allocation_count = 0u;
  std::uint64_t host_to_device_bytes = 0u;
  std::uint64_t device_to_host_bytes = 0u;
  const char *last_error = "ok";
};

[[nodiscard]] const char *
CpuSimdStrategyInfo(rund::kernel::CpuSimdStrategy strategy) noexcept;

[[nodiscard]] CpuProfile DetectCpu() noexcept;

[[nodiscard]] rund::kernel::CpuCaps
MakeCpuCaps(const CpuProfile &profile) noexcept;

[[nodiscard]] bool
CpuWindowsMatchPlan(const rund::kernel::ComputePlan &plan,
                    const rund::kernel::ComputeDispatchWindow *windows,
                    rund::kernel::u64 window_count) noexcept;

[[nodiscard]] bool
ExecuteCpu(void *context, const rund::kernel::ComputePlan &plan,
           const rund::kernel::LoweringArtifact &artifact,
           const rund::kernel::ComputeDispatchWindow *windows,
           rund::kernel::u64 window_count,
           const rund::kernel::BindingSet &bindings);

[[nodiscard]] bool ExecuteRetainedCpu(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &input,
    const rund::kernel::ExecutionMetadata &metadata,
    const rund::kernel::ComputeDispatchWindow *windows,
    rund::kernel::u64 window_count, const rund::kernel::BindingSet &bindings);

[[nodiscard]] const char *CpuLastError(void *context) noexcept;

} // namespace rund::node::accel::detail
