#pragma once

#include <accel/kernel/run/binding.hpp>

#include <cstdint>

namespace rund {

struct AccelRun {
  const AccelRunBinding *bindings = nullptr;
  std::uint64_t binding_count = 0u;
  std::uint64_t tile_count = 0u;
  bool fresh_evidence = false;
};

} // namespace rund
