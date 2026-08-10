#pragma once

#include "../state.hpp"

#include <cstdint>

namespace rund::compute::detail {

[[nodiscard]] Stats
begin_virtual_run_evidence(const VirtualPipelineState &state,
                           std::uint64_t active_count) noexcept;

[[nodiscard]] Status
publish_virtual_run_evidence(VirtualPipelineState &state, Stats stats,
                             Status status, std::uint64_t failed_page,
                             std::uint64_t output_hash,
                             bool poison_pipeline) noexcept;

} // namespace rund::compute::detail
