#pragma once

#include "registry/reservation.hpp"
#include "../../backend/phase.hpp"

#include "../../memory.hpp"
#include "../../publication.hpp"
#include "../../scratch.hpp"
#include "../../view.hpp"

#include <accel/api.hpp>
#include <accel/check.hpp>
#include <accel/kernel/value.hpp>

#include <kernel/core/checked.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>

namespace rund::node::accel::detail {

struct BackendRun;
struct KernelExecutionStep;
struct KernelExecution;
struct BackendOps;

// Shape and registry API form one declaration unit; reservations have their own value owner.
// clang-format off
#include "registry/shape.hpp"
#include "registry/api.hpp"
// clang-format on

} // namespace rund::node::accel::detail
