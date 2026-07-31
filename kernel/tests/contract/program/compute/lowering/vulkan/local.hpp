#pragma once

#include "contract/program/compute/lowering/support.hpp"
#include "contract/program/compute/lowering/fixed/nonlinear.hpp"
#include "test/assert.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/lowering/artifact/admission.hpp>
#include <kernel/program/compute/lowering/entry.hpp>
#include <kernel/program/compute/lowering/fusion/build.hpp>

#include <string_view>

namespace program_compute_contract {

int VulkanLoweringBase();
int VulkanLoweringFixedLane64();
int VulkanLoweringExpanded();
int VulkanLoweringScalarOps();
int VulkanLoweringBitOps();
int VulkanLoweringNonlinearOps();

} // namespace program_compute_contract
