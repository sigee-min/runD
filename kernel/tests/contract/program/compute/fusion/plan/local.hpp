#pragma once

#include "contract/program/compute/fusion/local.hpp"
#include "test/assert.hpp"

#include <string_view>

namespace program_compute_contract {

int RunFusionPlanSuccessContract();
int RunFusionPlanVisibilityContract();
int RunFusionPlanRejectContract();

} // namespace program_compute_contract
