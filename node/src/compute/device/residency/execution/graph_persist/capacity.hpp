#pragma once

#include <cstddef>
#include <limits>

namespace rund::compute::detail::residency::execution {

inline constexpr std::size_t GraphPersistCapacity = 32u;
inline constexpr std::size_t GraphPersistSlotCapacity = 2u;
static_assert(GraphPersistCapacity != 0u && GraphPersistSlotCapacity != 0u);
static_assert(GraphPersistCapacity <= std::numeric_limits<std::size_t>::max() /
                                          GraphPersistSlotCapacity);
inline constexpr std::size_t GraphPersistAggregateCapacity =
    GraphPersistCapacity * GraphPersistSlotCapacity;
static_assert(GraphPersistAggregateCapacity != 0u);

} // namespace rund::compute::detail::residency::execution
