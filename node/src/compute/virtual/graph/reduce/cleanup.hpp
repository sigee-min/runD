#pragma once

#include "model.hpp"

namespace rund::compute::detail::graph_reduce {

class FailLog;

// Terminally disposes every physical/logical owner still retained by one
// ticket. The caller separately stops active native execution and cancels the
// shared Forecast controller before invoking this per-ticket operation.
[[nodiscard]] bool cleanup_ticket(residency::Authority &, Ticket &,
                                  FailLog &) noexcept;

} // namespace rund::compute::detail::graph_reduce
