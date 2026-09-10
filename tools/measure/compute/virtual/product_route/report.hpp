#pragma once

#include "../product_route.hpp"

#include <cstdint>

namespace rund::measure::compute::virtual_residency {

void PrintProductRouteEvidence(const ProductRouteEvidence &,
                               std::uint64_t expected_runs);

} // namespace rund::measure::compute::virtual_residency
