#pragma once

#include <accel/api.hpp>
#include <accel/device.hpp>

#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/dsl.hpp>

#include <vector>

namespace node_accel_contract::runtime::window {

struct ResidentTileValue {
  rund::kernel::u32 value = 0u;
};

[[nodiscard]] bool PickUnavailableReasonIsPrecise(const rund::AccelDevice &pick,
                                                  rund::AccelApi api);

[[nodiscard]] const char *
BackendLastError(const rund::AccelDevice &pick) noexcept;

[[nodiscard]] std::vector<rund::kernel::ComputeDispatchWindow>
DispatchWindows(const rund::kernel::ComputePlan &plan);

} // namespace node_accel_contract::runtime::window
