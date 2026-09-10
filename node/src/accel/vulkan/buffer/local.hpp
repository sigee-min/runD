#pragma once

#include "../adapter/access.hpp"

#include <accel/device.hpp>

#include <cstdint>

namespace rund::node::accel::detail {

[[nodiscard]] std::uint64_t
VulkanBufferStorageBytes(const rund::AccelDevice &pick,
                         std::uint64_t logical_bytes) noexcept;

[[nodiscard]] std::uint64_t
VulkanPipelineTransferStorageBytes(const rund::AccelDevice &pick,
                                   std::uint64_t logical_bytes) noexcept;

} // namespace rund::node::accel::detail
