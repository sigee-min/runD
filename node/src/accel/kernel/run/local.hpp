#pragma once

#include <node/accel/context.hpp>

#include "../../context/internal.hpp"
#include "../bindings.hpp"
#include "../evidence.hpp"
#include "../execute.hpp"
#include "../plan.hpp"
#include <kernel/program/compute/graph/schema.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] inline bool RoleMatches(
    const rund::kernel::BufferRole expected,
    const rund::kernel::BufferRole actual) noexcept {
  return expected == actual &&
         (actual == rund::kernel::BufferRole::Read ||
          actual == rund::kernel::BufferRole::Write);
}

}  // namespace rund::node::accel::detail
