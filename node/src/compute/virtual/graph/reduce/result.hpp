#pragma once

#include "../reduce.hpp"

#include <cstdint>

namespace rund::compute::detail::graph_reduce {

[[nodiscard]] VirtualGraphResult failed(Status, std::uint64_t,
                                        bool poison = false) noexcept;

} // namespace rund::compute::detail::graph_reduce
