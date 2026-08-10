#pragma once

#include "../model.hpp"

#include <rund/compute/pipeline/memory.hpp>
#include <rund/compute/telemetry.hpp>

#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::measure::compute::virtual_residency {

void Seed(std::span<std::int32_t> values) noexcept;

[[nodiscard]] bool ValidOutput(std::span<const std::int32_t> values,
                               std::size_t active_count) noexcept;

[[nodiscard]] std::uint64_t ContentHash(std::span<const std::int32_t> values,
                                        std::size_t active_count) noexcept;

[[nodiscard]] bool ExactPreparationEvidence(
    const ::rund::compute::telemetry::Profile &preparation) noexcept;

[[nodiscard]] bool
ExactProfile(const ::rund::compute::telemetry::Profile &profile,
             const ::rund::compute::PipelinePlan &plan, Backend backend,
             std::size_t active_count, std::uint32_t sampled_runs) noexcept;

} // namespace rund::measure::compute::virtual_residency
