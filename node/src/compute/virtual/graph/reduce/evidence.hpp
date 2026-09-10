#pragma once

#include "model.hpp"

#include <span>

namespace rund::compute::detail::graph_reduce {

void classify_backing(Stats &, std::span<const residency::PageUse>,
                      std::uint64_t fetched, bool speculative,
                      bool accelerator) noexcept;
[[nodiscard]] bool record_input_evidence(Stats &, const VirtualRunProjection &,
                                         const Ticket &,
                                         std::uint64_t fetched_pages,
                                         std::uint64_t backing_bytes) noexcept;

} // namespace rund::compute::detail::graph_reduce
