#pragma once

#include <accel/api.hpp>
#include <accel/check.hpp>
#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/cpu.hpp>
#include <kernel/program/compute/model.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace rund {

struct AccelPolicy {
  AccelApi preferred[3] = {AccelApi::Metal, AccelApi::Vulkan, AccelApi::Fake};
  std::uint32_t preferred_count = 2u;
  bool allow_fake = false;
};

struct AccelBackendInfo {
  std::string device_name{};
  std::string driver_name{};
  std::string driver_info{};
  std::uint64_t storage_alignment{};
  std::uint64_t storage_bytes{};
};

struct AccelDevice {
  AccelCheck check{};
  AccelApi api = AccelApi::Auto;
  rund::kernel::ComputeCaps caps{};
  rund::kernel::CpuCaps cpu_caps{};
  rund::kernel::ComputeBackendDispatch backend{};
  AccelBackendInfo backend_info{};
  std::shared_ptr<void> owner{};
};

} // namespace rund
