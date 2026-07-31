#pragma once

#include <rund/compute/flow/model.hpp>

namespace rund::compute {

template <class T, class Card> class StageRef final {
// clang-format off
#define RUND_COMPUTE_FLOW_STAGE_MEMBERS
#include <rund/compute/flow/stage/solve.hpp>
#include <rund/compute/flow/stage/filter.hpp>
#include <rund/compute/flow/stage/expand.hpp>
#include <rund/compute/flow/stage/window.hpp>
#include <rund/compute/flow/stage/collect.hpp>
#include <rund/compute/flow/stage/state.hpp>
#undef RUND_COMPUTE_FLOW_STAGE_MEMBERS
// clang-format on
};

} // namespace rund::compute
