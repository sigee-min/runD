#pragma once

#include "model.hpp"

#include <span>

namespace rund::compute::detail::graph_reduce {

void classify_backing(Stats &, std::span<const residency::PageUse>,
                      std::uint64_t fetched, bool speculative,
                      bool accelerator) noexcept;
[[nodiscard]] bool
record_input_evidence(Stats &, const VirtualRunProjection &, Backend,
                      std::span<const residency::GraphLeasePort>,
                      std::span<const residency::CacheBinding>,
                      std::span<const residency::CacheTransition>,
                      std::uint64_t fetched_pages,
                      std::uint64_t backing_bytes) noexcept;

} // namespace rund::compute::detail::graph_reduce
