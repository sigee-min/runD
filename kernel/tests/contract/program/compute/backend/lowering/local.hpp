#pragma once

#include "contract/program/compute/lowering/fixed/nonlinear.hpp"
#include "contract/program/compute/lowering/support.hpp"

#include <kernel/program/compute/binding/model.hpp>
#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/lowering/artifact/admission.hpp>
#include <kernel/program/compute/lowering/entry.hpp>
#include <kernel/program/compute/lowering/fusion/build.hpp>

#include <string_view>
#include <utility>

namespace program_compute_contract {

int RunComputeBackendLoweringBaseContract();
int RunComputeBackendLoweringMetalContract();
int RunComputeBackendLoweringNameContract();
int RunComputeBackendLoweringRejectContract();
int RunComputeBackendLoweringArtifactContract();
int RunComputeBackendLoweringRuntimeContract();
int RunComputeVulkanLoweringContract();

namespace backend_lowering_support {

using namespace lowering_support;
using namespace nonlinear_support;
using rund::kernel::i32;

#include "local/ops.hpp"

} // namespace backend_lowering_support
} // namespace program_compute_contract
