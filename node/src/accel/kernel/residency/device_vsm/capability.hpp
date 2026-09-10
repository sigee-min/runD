#pragma once

#include <accel/check.hpp>

#include <cstdint>

namespace rund::node::accel::detail {

struct DeviceVsmCapability final {
  rund::AccelCheck check{false, "accel_kernel_pipeline_invalid"};
  std::uint64_t retained_bytes{};
  std::uint64_t transient_bytes{};
  std::uint8_t width{};
  bool gpu_addressable_backing{};
  bool device_generated_recurrence{};
  bool fixed_native_storage{};
  bool fixed_common_storage{};
  bool one_native_submit{};
  bool host_service_turns_zero{};
  bool host_epoch_callbacks_zero{};
  bool aggregate_terminal_once{};
  bool bounded_page_io{};
  bool physical_ring_storage{};
};

} // namespace rund::node::accel::detail
