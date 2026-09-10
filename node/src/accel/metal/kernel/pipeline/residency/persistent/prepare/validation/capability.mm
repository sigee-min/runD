#include "../../../../../../../kernel/prepared/interface/api.hpp"
#include "../../../../../../../kernel/prepared/model.hpp"
#include "../../internal.hpp"

#include <array>

namespace rund::node::accel::detail {

PersistentResidencySlidingCapability
QueryMetalPreparedPersistentSlidingCapability(
    const std::span<const PreparedResidencyPersistentSlidingRole> roles,
    const std::uint64_t coordinate_count, const ResidencySlidingMemory memory,
    const PersistentResidencySlidingMode mode) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  if (roles.size() != 2u && roles.size() != 4u) {
    return PersistentResidencySlidingCapability{
        .check = {false, "compute_backend_unsupported"},
        .memory = memory,
        .mode = mode};
  }
  std::array<metal_persistent_sliding::NativeRole,
             PersistentResidencySlidingCapacity>
      native{};
  for (std::size_t slot = 0u; slot < roles.size(); ++slot) {
    native[slot] = metal_persistent_sliding::native_role(roles[slot]);
  }
  MetalAdapter *adapter = nullptr;
  const std::uint64_t issue = metal_persistent_sliding::first_invalid_structure(
      std::span<const metal_persistent_sliding::NativeRole>{native},
      roles.size(), coordinate_count, memory, mode, adapter);
  if (issue != metal_persistent_sliding::request_issue_key(
                   metal_persistent_sliding::RequestIssue::Valid)) {
    return PersistentResidencySlidingCapability{
        .check =
            issue ==
                    metal_persistent_sliding::request_issue_key(
                        metal_persistent_sliding::RequestIssue::GenerationRange)
                ? rund::AccelCheck{false, "compute_pipeline_capacity"}
                : rund::AccelCheck{false, "accel_kernel_pipeline_invalid"},
        .memory = memory,
        .mode = mode};
  }
  return PersistentResidencySlidingCapability{
      .check = {true, "ok"},
      .memory = memory,
      .width = static_cast<std::uint8_t>(roles.size()),
      .whole_run_preencoded = true,
      .host_epoch_callbacks_zero = true,
      .mode = mode,
  };
#else
  (void)roles;
  (void)coordinate_count;
  return PersistentResidencySlidingCapability{
      .check = {false, "compute_backend_unsupported"},
      .memory = memory,
      .mode = mode};
#endif
}

} // namespace rund::node::accel::detail
