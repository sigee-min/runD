#pragma once

#include "../../manifest.hpp"

namespace rund::kernel {
struct ComputePlan;
}

namespace rund::node::accel::detail {

[[nodiscard]] bool
AddMetalStepSourceRecipes(const KernelExecutionStep &step,
                          const rund::kernel::ComputePlan &plan,
                          bool controlled, bool has_checks,
                          PreparedBackendManifest &manifest) noexcept;

} // namespace rund::node::accel::detail
