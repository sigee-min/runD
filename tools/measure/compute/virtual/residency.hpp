#pragma once

#include "../model.hpp"

namespace rund::measure::compute {

enum class VirtualResidencyBacking : std::uint8_t { Callback, Resident };

void PrintVirtualResidencyColumns();
[[nodiscard]] bool MeasureVirtualResidency(Backend backend,
                                           VirtualResidencyBacking backing);

} // namespace rund::measure::compute
