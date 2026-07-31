#include <accel/api.hpp>
#include <accel/check.hpp>
#include <accel/device.hpp>

#include "local.hpp"

#include <memory>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] rund::AccelDevice RejectCpu(const char *const reason) {
  return rund::AccelDevice{
      .check = rund::AccelCheck{false, reason},
      .api = rund::AccelApi::Cpu,
      .cpu_caps = rund::kernel::CpuCaps{.reason = reason},
  };
}

} // namespace

rund::AccelDevice PickCpu() {
  const CpuProfile profile = DetectCpu();
  const rund::kernel::CpuCaps caps = MakeCpuCaps(profile);
  if (!caps.ok) {
    return RejectCpu(caps.reason);
  }

  std::shared_ptr<CpuAdapter> adapter = std::make_shared<CpuAdapter>();
  adapter->caps = caps;
  adapter->generic_caps = rund::kernel::ComputeCaps{
      .api = rund::kernel::ComputeApi::Cpu,
      .device_bytes = 64u * 1024u * 1024u,
      .staging_bytes = 64u * 1024u * 1024u,
      .max_window_tiles = 64u * 1024u,
      .storage_alignment = 64u,
      .subgroup_width = caps.fixed_lane32_lanes,
      .ok = true,
      .reason = "ok",
  };
  adapter->info = rund::AccelBackendInfo{
      .device_name = "cpu",
      .driver_name = profile.source,
      .driver_info = CpuSimdStrategyInfo(caps.strategy),
      .storage_alignment = adapter->generic_caps.storage_alignment,
      .storage_bytes = adapter->generic_caps.device_bytes,
  };
  std::shared_ptr<void> owner = adapter;
  adapter->owner_token = owner;
  return rund::AccelDevice{
      .check = rund::AccelCheck{true, "ok"},
      .api = rund::AccelApi::Cpu,
      .caps = adapter->generic_caps,
      .cpu_caps = adapter->caps,
      .backend =
          rund::kernel::ComputeBackendDispatch{
              .context = adapter.get(),
              .execute = ExecuteCpu,
              .last_error = CpuLastError,
          },
      .backend_info = adapter->info,
      .owner = owner,
  };
}

} // namespace rund::node::accel::detail
