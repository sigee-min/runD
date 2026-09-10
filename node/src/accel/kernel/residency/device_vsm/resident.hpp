#pragma once

#include <accel/context/buffer.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace rund::node::accel::detail {

inline constexpr std::size_t DeviceVsmResidentCapacity = 8u;

enum class DeviceVsmResidentRole : std::uint8_t { Input, Output };

struct DeviceVsmResidentBinding final {
  DeviceVsmResidentRole role{DeviceVsmResidentRole::Input};
  rund::kernel::ResidentBufferRef backing{};
  std::shared_ptr<void> handle{};
};

struct DeviceVsmResidentSet final {
  std::array<DeviceVsmResidentBinding, DeviceVsmResidentCapacity> rows{};
  std::uint32_t count{};
  std::uint32_t input_count{};
  std::uint32_t output_count{};
};

[[nodiscard]] inline DeviceVsmResidentSet
device_vsm_resident_pair(const rund::kernel::ResidentBufferRef &input,
                         std::shared_ptr<void> input_handle,
                         const rund::kernel::ResidentBufferRef &output,
                         std::shared_ptr<void> output_handle) noexcept {
  DeviceVsmResidentSet result{};
  result.rows[0u] = DeviceVsmResidentBinding{
      .role = DeviceVsmResidentRole::Input,
      .backing = input,
      .handle = std::move(input_handle),
  };
  result.rows[1u] = DeviceVsmResidentBinding{
      .role = DeviceVsmResidentRole::Output,
      .backing = output,
      .handle = std::move(output_handle),
  };
  result.count = 2u;
  result.input_count = 1u;
  result.output_count = 1u;
  return result;
}

} // namespace rund::node::accel::detail
