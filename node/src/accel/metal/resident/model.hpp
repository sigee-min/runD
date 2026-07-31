#pragma once

#include <accel/check.hpp>
#include <kernel/program/compute/binding/model.hpp>

#include "../../resident/model.hpp"

#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

struct MetalResidentBuffer : ResidentEntry {
  std::weak_ptr<void> device_buffer{};
};

struct MetalResidentBufferResult {
  rund::AccelCheck check{};
  rund::kernel::ResidentBufferRef ref{};
  std::shared_ptr<void> handle{};
  std::shared_ptr<void> device_buffer{};
};

} // namespace rund::node::accel::detail
