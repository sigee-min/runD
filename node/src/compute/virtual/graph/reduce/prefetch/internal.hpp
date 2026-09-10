#pragma once

#include "../prefetch.hpp"

#include "../../../../pipeline/run/clock.hpp"
#include "../../../run/backing.hpp"
#include "../../../run/projection.hpp"
#include "../../../state.hpp"
#include "../../../stats.hpp"
#include "../../forecast.hpp"
#include "../authority.hpp"
#include "../evidence.hpp"
#include "../projection.hpp"
#include "../timeline.hpp"
#include "../../../../device/residency/registry/graph_forecast_owner.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <span>
#include <utility>

namespace rund::compute::detail::graph_reduce::prefetch_detail {

using ::rund::detail::counter::Accumulate;

} // namespace rund::compute::detail::graph_reduce::prefetch_detail
