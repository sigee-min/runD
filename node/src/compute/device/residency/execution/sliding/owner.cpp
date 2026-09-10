#include "internal.hpp"

namespace rund::compute::detail::residency::execution {

std::atomic<std::uint64_t> NextSlidingOwner{1u};

} // namespace rund::compute::detail::residency::execution
