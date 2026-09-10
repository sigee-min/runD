#pragma once

#include "identity.hpp"

#include "../../prepared/template/registry.hpp"
#include "../run.hpp"

namespace rund::node::accel::detail::backend_template_plan {

[[nodiscard]] bool same_windows(const DispatchWindowStorage &left,
                                const DispatchWindowStorage &right) noexcept;

[[nodiscard]] bool same_layout(const KernelViewLayout *left,
                               const KernelViewLayout *right) noexcept;

[[nodiscard]] bool same_layout(const KernelScratchLayout *left,
                               const KernelScratchLayout *right) noexcept;

[[nodiscard]] bool
same_map_binding_identity(const PreparedKernelProgramBindingIdentity &left,
                          const PreparedKernelProgramBindingIdentity &right,
                          std::uint64_t alignment,
                          rund::kernel::ComputeApi api) noexcept;

[[nodiscard]] bool same_program_map_specialization(
    const KernelExecution &execution, const PreparedKernelProgramRoute &left,
    const PreparedKernelProgramRoute &right, std::uint64_t alignment) noexcept;

[[nodiscard]] bool
same_ref_layout(const rund::kernel::ResidentBindingRange &left,
                const rund::kernel::ResidentBindingRange &right,
                std::uint64_t alignment, rund::kernel::ComputeApi api) noexcept;

[[nodiscard]] bool same_map_layout(const BoundStep &left,
                                   const BoundStep &right,
                                   std::uint64_t alignment,
                                   rund::kernel::ComputeApi api) noexcept;

[[nodiscard]] std::uint64_t
primitive_pass_count(const KernelExecutionStep &step) noexcept;

} // namespace rund::node::accel::detail::backend_template_plan
