#pragma once

#include "contract/program/compute/graph/local.hpp"

#include <string_view>

namespace program_compute_contract {

int GraphRejectPrimitive();
int GraphRejectNode();
int GraphRejectBuffer();
int GraphRejectNumericPolicy();

} // namespace program_compute_contract
