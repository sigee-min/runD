#pragma once

#include "model.hpp"

#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace rund::node::accel::detail {

struct MetalResidentState final {
  std::mutex mutex{};
  std::unordered_map<std::uint64_t, MetalResidentBuffer> buffers{};
  std::uint64_t next_id = 1u;
};

} // namespace rund::node::accel::detail
