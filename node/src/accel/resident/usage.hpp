#pragma once

#include "model.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] inline bool
KnownResidentUsage(const rund::kernel::u32 usage) noexcept {
  return usage == rund::kernel::kResidentUsageRead ||
         usage == rund::kernel::kResidentUsageWrite;
}

[[nodiscard]] inline bool ReadCapable(const ResidentDesc &desc) noexcept {
  return desc.read_capable || desc.usage == rund::kernel::kResidentUsageRead;
}

[[nodiscard]] inline bool WriteCapable(const ResidentDesc &desc) noexcept {
  return desc.write_capable || desc.usage == rund::kernel::kResidentUsageWrite;
}

} // namespace rund::node::accel::detail
