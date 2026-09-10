#pragma once

#include "../../internal.hpp"

namespace rund::compute::detail::residency::physical::promote {

struct Projection final {
  execution::Node input{};
  execution::Node output{};
  execution::WindowFootprintProjection footprint{};
  FrameRegion device_input{};
  FrameRegion device_output{};
  FrameRegion host_output{};
  bool assembles_window{};
};

struct Selection final {
  std::array<std::uint32_t, execution::WindowFootprintSourceCapacity>
      input_slots{};
  std::array<std::uint32_t, execution::UseCapacity> output_slots{};
  std::array<std::uint32_t, execution::WindowFootprintSourceCapacity>
      host_frames{};
  std::array<std::uint32_t, execution::UseCapacity> device_input_frames{};
  std::array<std::uint32_t, execution::UseCapacity> device_output_frames{};
  std::array<std::uint32_t, execution::UseCapacity> host_output_frames{};
  std::uint64_t transfer_bytes{};
  std::uint32_t transfer_mask{};
};

struct Commit final {
  std::uint64_t turn{};
  std::uint32_t source_count{};
  std::uint32_t input_count{};
  std::uint32_t output_count{};
};

} // namespace rund::compute::detail::residency::physical::promote
