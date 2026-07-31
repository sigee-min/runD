#pragma once

#include <accel/api.hpp>
#include <accel/kernel/check.hpp>
#include <kernel/program/compute/model.hpp>

#include <cstdint>
#include <memory>

namespace rund {

struct AccelKernel {
  AccelKernelCheck check{};
  std::uint64_t kernel_id = 0u;
  std::uint64_t graph_id_hi = 0u;
  std::uint64_t graph_id_lo = 0u;
  std::uint64_t node_count = 0u;
  AccelApi api = AccelApi::Auto;
  rund::kernel::ComputeScalar scalar = rund::kernel::ComputeScalar::Lane32;
  rund::kernel::ComputeDomain domain = rund::kernel::ComputeDomain::Fixed;
  rund::kernel::ComputeCaps frozen_caps{};
  std::uint64_t context_id = 0u;
  std::shared_ptr<void> owner{};
  const char* reason = "accel_kernel_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return check.ok;
  }
};

}  // namespace rund
