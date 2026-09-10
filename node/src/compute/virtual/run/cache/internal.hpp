#pragma once

#include "../../../device/residency/pool.hpp"
#include "../../../pipeline/local.hpp"
#include "../../../pipeline/residency/authority.hpp"
#include "../../../pipeline/run/clock.hpp"
#include "../../../pipeline/transfer.hpp"
#include "../../backing.hpp"
#include "../backing.hpp"
#include "../cache.hpp"
#include "../transaction.hpp"

#include <cstddef>
#include <cstdint>

namespace rund::compute::detail::virtual_cache_detail {

// The frame calculation is shared by output admission and retention checks;
// its definition remains in the output owner so no second frame predicate can
// drift from the physical Pool layout.
[[nodiscard]] bool output_bank_frame(const PipelineState &,
                                     const VirtualRunProjection &,
                                     std::uint32_t frame,
                                     std::size_t &local) noexcept;

} // namespace rund::compute::detail::virtual_cache_detail
