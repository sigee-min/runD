#pragma once

#include "model.hpp"

namespace rund::measure::compute::preparation_memory {

void PrintObservation(Backend backend, bool materialize,
                      const PreparationMemoryObservation &observed);

} // namespace rund::measure::compute::preparation_memory
