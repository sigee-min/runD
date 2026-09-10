#pragma once

#include "phase.hpp"
#include "../coordinate.hpp"

#include "../bindings/build.hpp"
#include "../plan.hpp"
#include "../publication.hpp"
#include "../reset/model.hpp"
#include "../schedule.hpp"
#include "../scratch.hpp"
#include "../storage.hpp"
#include "../view.hpp"

#include <accel/check.hpp>
#include <accel/context/value.hpp>
#include <accel/runtime.hpp>
#include <rund/compute/pipeline/coordinate.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace rund::node::accel::detail {

struct BackendOps;
struct PreparedKernelTemplateRegistry;

// These fragments form one declaration unit. Keep dependencies before users.
// clang-format off
#include "run/bound.hpp"
#include "run/model.hpp"
#include "run/window.hpp"
#include "run/batch.hpp"
#include "run/binding.hpp"
// clang-format on

} // namespace rund::node::accel::detail
