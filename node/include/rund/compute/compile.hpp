#pragma once

#include <cstddef>
#include <cstdint>

namespace rund::compute {

struct Compile final {
  std::uint32_t workers = 2u;
  std::size_t capacity = 64u;
};

} // namespace rund::compute
