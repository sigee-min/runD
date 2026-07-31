#include "info.hpp"

#include "state.hpp"

#include <rund/compute/device.hpp>

#include <charconv>
#include <new>
#include <string>
#include <string_view>
#include <utility>

namespace rund::compute::detail {
namespace {

[[nodiscard]] constexpr std::string_view
strategy_name(const kernel::CpuSimdStrategy strategy) noexcept {
  switch (strategy) {
  case kernel::CpuSimdStrategy::Scalar:
    return "scalar";
  case kernel::CpuSimdStrategy::Sse2:
    return "sse2";
  case kernel::CpuSimdStrategy::Avx2:
    return "avx2";
  case kernel::CpuSimdStrategy::Avx512:
    return "avx512";
  case kernel::CpuSimdStrategy::Neon:
    return "neon";
  }
  return {};
}

void append(std::string &text, const std::string_view label,
            const std::uint32_t value) {
  text.append(label);
  char digits[10]{};
  const auto converted = std::to_chars(digits, digits + sizeof(digits), value);
  text.append(digits, static_cast<std::size_t>(converted.ptr - digits));
}

[[nodiscard]] DeviceInfo cpu_info(const DeviceState &state,
                                  const CpuDeviceState &cpu) {
  const std::string_view strategy = strategy_name(cpu.caps.strategy);

  std::string name{"cpu/"};
  name.append(strategy);

  std::string details;
  details.reserve(100u);
  append(details, "workers=", cpu.workers.requested_worker_width);
  append(details, ";lane:bytes=", cpu.caps.lane_bytes);
  append(details, ";fixed:lane32:lanes=", cpu.caps.fixed_lane32_lanes);
  append(details, ";fixed:lane64:lanes=", cpu.caps.fixed_lane64_lanes);

  return DeviceInfo{
      .backend = state.backend,
      .name = std::move(name),
      .driver = "rund/built-in-pool",
      .driver_details = std::move(details),
      .storage_alignment = 64u,
      .storage_bytes = 64u * 1024u * 1024u,
  };
}

[[nodiscard]] Result<DeviceInfo> invalid() noexcept {
  return Result<DeviceInfo>::fail(Reason::DeviceInfoInvalid);
}

} // namespace

Result<DeviceInfo>
snapshot_device_info(const std::shared_ptr<DeviceState> &state) noexcept {
  if (state == nullptr) {
    return invalid();
  }

  try {
    switch (state->backend) {
    case Backend::Unavailable:
      return invalid();
    case Backend::Cpu: {
      const CpuDeviceState *const cpu = cpu_device(*state);
      if (cpu == nullptr || !cpu->caps || !cpu->workers ||
          strategy_name(cpu->caps.strategy).empty()) {
        return invalid();
      }
      return Result<DeviceInfo>::success(cpu_info(*state, *cpu));
    }
    case Backend::Metal:
    case Backend::Vulkan: {
      const AccelDeviceState *const accel = accel_device(*state);
      if (accel == nullptr) {
        return invalid();
      }
      const rund::AccelBackendInfo &source = accel->pick.backend_info;
      return Result<DeviceInfo>::success(DeviceInfo{
          .backend = state->backend,
          .name = source.device_name,
          .driver = source.driver_name,
          .driver_details = source.driver_info,
          .storage_alignment = source.storage_alignment,
          .storage_bytes = source.storage_bytes,
      });
    }
    }
    return invalid();
  } catch (const std::bad_alloc &) {
    return Result<DeviceInfo>::fail(Reason::DeviceInfoCapacity);
  }
}

} // namespace rund::compute::detail

namespace rund::compute {

Result<DeviceInfo> Device::info() const noexcept {
  return detail::snapshot_device_info(state_);
}

} // namespace rund::compute
