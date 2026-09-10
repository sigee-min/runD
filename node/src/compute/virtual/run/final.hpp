#pragma once

#include "backing.hpp"
#include "backing_set.hpp"
#include "dispatch.hpp"
#include "reduce.hpp"
#include "transaction.hpp"

#include "../state.hpp"

#include <rund/compute/stats.hpp>

namespace rund::compute::detail {

// Closes the optional output transaction before publishing one run evidence
// record. This is the sole coordinator terminal for transaction status and
// public Pipeline evidence.
[[nodiscard]] Status
finish_virtual_run(VirtualPipelineState &, Stats, VirtualRunTransaction &,
                   Status, VirtualRunWriteCertainty, std::uint64_t failed_page,
                   std::uint64_t output_hash, bool &poison_pipeline,
                   VirtualRunResources *resources = nullptr) noexcept;

[[nodiscard]] Status finish_virtual_empty(VirtualPipelineState &, Stats,
                                          VirtualRunTransaction &,
                                          const VirtualRunProjection &,
                                          VirtualBacking &, VirtualReduction &,
                                          std::uint64_t &failed_page,
                                          bool &poison_pipeline,
                                          VirtualRunResources *resources = nullptr) noexcept;

// Owns completion of the lower epoch/reduction result before the shared
// transaction and evidence terminal. Direct and Graph owners enter through
// finish_virtual_run without repeating this work.
[[nodiscard]] Status
finish_virtual_dispatch(VirtualPipelineState &, Stats, VirtualRunTransaction &,
                        const VirtualRunProjection &, VirtualBacking &,
                        VirtualReduction &, const VirtualRunDispatchResult &,
                        std::uint64_t &failed_page,
                        bool &poison_pipeline,
                        VirtualRunResources *resources = nullptr) noexcept;

} // namespace rund::compute::detail
