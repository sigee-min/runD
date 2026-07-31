#pragma once

#include <accel/device.hpp>

#include "../primitive/local.hpp"

#include <array>

namespace node_accel_contract {

[[nodiscard]] bool SortIdentityU32MatchesCpuReference(
    const rund::AccelDevice &pick, rund::kernel::ComputeScalar scalar,
    const std::array<rund::kernel::u32, 8u> &input_keys,
    rund::kernel::u32 key_bits = 0u);

} // namespace node_accel_contract
