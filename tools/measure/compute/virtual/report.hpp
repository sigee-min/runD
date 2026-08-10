#pragma once

#include "../model.hpp"
#include "model.hpp"

#include <rund/compute/pipeline/memory.hpp>
#include <rund/compute/stats.hpp>
#include <rund/compute/telemetry.hpp>

#include <cstddef>
#include <cstdint>

namespace rund::measure::compute::virtual_residency {

void PrintColumns();

void PrintEvidence(const char *phase, Backend backend, std::size_t active_count,
                   const WallEvidence &wall, std::uint32_t first_value_bits,
                   const ::rund::compute::telemetry::Profile &profile,
                   const ::rund::compute::PipelinePlan &plan,
                   const ::rund::compute::telemetry::Profile &preparation,
                   const ObservationEvidence &observation);

} // namespace rund::measure::compute::virtual_residency
