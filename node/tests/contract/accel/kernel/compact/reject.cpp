#include <accel/api.hpp>
#include <accel/device.hpp>
#include <accel/kernel/evidence.hpp>
#include <kernel/program/compute/compact/reference.hpp>

#include "local.hpp"

namespace node_accel_contract {

bool CompactRejectsCapacityInsufficient(
    const rund::AccelDevice &pick, const rund::kernel::ComputeScalar scalar) {
  if (!pick.check.ok) {
    return false;
  }
  const std::array<rund::kernel::u32, 8u> flags{1u, 0u, 1u, 7u, 0u, 1u, 0u, 1u};
  const std::uint64_t capacity = 3u;
  std::array<rund::kernel::u32, 8u> ignored{};
  std::uint64_t ignored_count = 0u;
  if (rund::kernel::ReferenceCompactIdsU32(flags.data(), flags.size(), capacity,
                                           ignored.data(), &ignored_count)
          .ok) {
    return false;
  }
  const compact::RunContext ctx =
      compact::Compile(pick, scalar, flags, capacity);
  if (!ctx.valid) {
    return false;
  }
  const rund::AccelEvidence evidence = compact::Run(ctx, flags.size());
  const bool physical_dispatches = evidence.dispatch_count == 0u;
  return primitive::EvidenceReason(evidence,
                                   "compute_compact_capacity_insufficient") &&
         physical_dispatches &&
         evidence.original_dispatch_count == ctx.plan.pass_count &&
         evidence.final_dispatch_count == ctx.plan.pass_count &&
         evidence.host_to_device_bytes == 0u &&
         evidence.device_to_host_bytes == 0u;
}

} // namespace node_accel_contract
