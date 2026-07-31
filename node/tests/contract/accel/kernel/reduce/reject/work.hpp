#pragma once

#include <kernel/program/compute/reduce/reference.hpp>

#include "../local.hpp"

#include <array>

namespace node_accel_contract::reduce::reject {

struct Work {
  std::array<rund::kernel::u32, 2u> input{0xffffffffu, 1u};
};

[[nodiscard]] inline bool ReferenceRejectsOverflow(const Work &work) {
  rund::kernel::u32 expected = 0u;
  return !rund::kernel::ReferenceReduceSumU32(work.input.data(), &expected,
                                              work.input.size())
              .ok;
}

} // namespace node_accel_contract::reduce::reject
