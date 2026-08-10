#pragma once

#include "../model.hpp"

namespace rund::measure::compute {

void PrintVirtualResidencyColumns();
[[nodiscard]] bool MeasureVirtualResidency(Backend backend);

} // namespace rund::measure::compute
