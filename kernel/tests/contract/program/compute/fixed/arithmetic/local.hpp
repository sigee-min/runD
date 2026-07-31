#pragma once

#include "contract/program/compute/lowering/support.hpp"
#include "test/assert.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/lowering/artifact/admission.hpp>
#include <kernel/program/compute/lowering/entry.hpp>
#include <kernel/program/compute/lowering/fusion/build.hpp>

#include <array>
#include <string>
#include <string_view>

namespace program_compute_contract {

int RunComputeFixedArithmeticDslContract();
int RunComputeFixedArithmeticLoweringContract();
int RunComputeFixedArithmeticRejectContract();
int RunComputeFixedArithmeticFusionContract();

} // namespace program_compute_contract
