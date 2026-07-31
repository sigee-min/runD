#pragma once

#include <rund/compute/backend.hpp>

#include <cstdint>
#include <string>

namespace rund::compute {

struct DeviceInfo final {
  Backend backend{Backend::Cpu};
  std::string name;
  std::string driver;
  std::string driver_details;
  std::uint64_t storage_alignment{};
  std::uint64_t storage_bytes{};

  [[nodiscard]] bool operator==(const DeviceInfo &) const = default;
};

} // namespace rund::compute
