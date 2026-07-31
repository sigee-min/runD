#pragma once

#include <kernel/program/compute/binding/model.hpp>

#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

struct ResidentDesc final {
  rund::kernel::u64 bytes = 0u;
  rund::kernel::u64 element_bytes = 0u;
  rund::kernel::u64 stride_bytes = 0u;
  rund::kernel::u64 count = 0u;
  rund::kernel::u32 usage = 0u;
  bool read_capable = false;
  bool write_capable = false;
};

struct ResidentEntry {
  std::uint64_t id = 0u;
  rund::kernel::u64 bytes = 0u;
  rund::kernel::u64 element_bytes = 0u;
  rund::kernel::u64 stride_bytes = 0u;
  rund::kernel::u64 count = 0u;
  rund::kernel::u32 usage = 0u;
  bool read_capable = false;
  bool write_capable = false;
  std::weak_ptr<void> owner{};
};

} // namespace rund::node::accel::detail
