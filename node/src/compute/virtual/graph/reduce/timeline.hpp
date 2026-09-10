#pragma once

#include "model.hpp"

#include "../../stats.hpp"

#include <cstdint>
#include <optional>

namespace rund::compute::detail::graph_reduce {

[[nodiscard]] std::uint64_t duration(Interval) noexcept;
[[nodiscard]] bool add_concurrent(Timeline &, Interval) noexcept;
[[nodiscard]] bool add_transfer(Timeline &, Interval,
                                Timeline::Direction) noexcept;
[[nodiscard]] bool add_timeline(Timeline &, Interval,
                                std::optional<Timeline::Direction>) noexcept;
[[nodiscard]] bool record_interval(Timeline *, Interval,
                                   std::optional<Timeline::Direction>,
                                   ResidencyStats &) noexcept;
void observe_timeline(Timeline &, const residency::ExecutionReceipt &,
                      ResidencyStats &) noexcept;

} // namespace rund::compute::detail::graph_reduce
