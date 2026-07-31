#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace rund::node::accel::detail {

struct MetalPipelineIndex final {
  std::unordered_multimap<std::uint64_t, std::size_t> entries{};
  std::unordered_multimap<std::uint64_t, std::size_t> named{};
};

} // namespace rund::node::accel::detail
