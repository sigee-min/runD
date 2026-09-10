#pragma once

#include <cstdint>

namespace rund::node::accel::detail {

struct BackendRun;
struct KernelExecution;
struct PreparedKernelProgramRoute;

} // namespace rund::node::accel::detail

namespace rund::kernel {

struct ComputePlan;

} // namespace rund::kernel

namespace rund::node::accel::detail::backend_template_plan {

struct MapSpecializationFingerprint final {
  std::uint64_t hi{0x72756e442e6d6170ull};
  std::uint64_t lo{0x2e7370656369616cull};
  bool ok{};
};

[[nodiscard]] bool same_plan(const rund::kernel::ComputePlan &left,
                             const rund::kernel::ComputePlan &right) noexcept;

[[nodiscard]] MapSpecializationFingerprint
program_map_specialization_fingerprint(
    const KernelExecution &execution,
    const PreparedKernelProgramRoute &route) noexcept;

[[nodiscard]] MapSpecializationFingerprint
runtime_map_specialization_fingerprint(const BackendRun &run) noexcept;

[[nodiscard]] bool
same_program_template(const KernelExecution &execution,
                      const PreparedKernelProgramRoute &left,
                      const PreparedKernelProgramRoute &right,
                      std::uint64_t storage_alignment) noexcept;

[[nodiscard]] bool same_template(const BackendRun &left,
                                 const BackendRun &right,
                                 std::uint64_t storage_alignment) noexcept;

} // namespace rund::node::accel::detail::backend_template_plan
