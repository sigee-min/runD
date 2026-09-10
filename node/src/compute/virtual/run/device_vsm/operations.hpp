#pragma once

#include "shape.hpp"

#include <memory>
#include <span>

namespace rund::compute::detail::residency {
class Pool;
}

namespace rund::compute::detail {
struct VirtualRunResources;
class VirtualDeviceVsmRouteProof;
}

namespace rund::compute::detail::device_vsm_product_detail {

[[nodiscard]] inline bool
owner_lost(const std::shared_ptr<DeviceVsmProductOwner> &owner) noexcept {
  return owner == nullptr || owner->quarantine != nullptr ||
         (owner->evidence != nullptr && owner->evidence->quarantined);
}

inline void
quarantine_owner(const std::shared_ptr<DeviceVsmProductOwner> &owner) noexcept {
  if (owner == nullptr) {
    return;
  }
  if (owner->evidence != nullptr) {
    owner->evidence->quarantined = true;
  }
  owner->quarantine = owner;
}

[[nodiscard]] inline bool
lifecycle_open(const DeviceVsmProductRun &run) noexcept {
  return run.registration != nullptr && run.lease.has_value() &&
         static_cast<bool>(*run.lease) && run.backing_recovery;
}

[[nodiscard]] ::rund::AccelCheck admit_virtual_device_vsm(
    const VirtualPipelineState &, const VirtualRunProjection &,
    const VirtualDeviceVsmRouteProof &) noexcept;
[[nodiscard]] bool select_pipelines(
    VirtualPipelineState &, VirtualRunTopology,
    std::array<std::shared_ptr<PipelineState>, DeviceVsmPipelineCapacity> &,
    std::array<std::uint32_t, DeviceVsmPipelineCapacity> &,
    std::size_t &) noexcept;
[[nodiscard]] node::accel::detail::DeviceVsmIdentity
route_stamp(const VirtualPipelineState &, const VirtualRunProjection &,
            const VirtualDeviceVsmRouteProof &) noexcept;
[[nodiscard]] bool proof_matches(const VirtualPipelineState &,
                                 const VirtualRunProjection &,
                                 const VirtualDeviceVsmRouteProof &) noexcept;
[[nodiscard]] std::shared_ptr<DeviceVsmProductOwner>
prepare_cold_owner(VirtualPipelineState &, const VirtualRunProjection &,
                   const AccelDeviceState &,
                   const node::accel::detail::DeviceVsmIdentity &,
                   const VirtualDeviceVsmRouteProof &, Status &,
                   const char *&) noexcept;
[[nodiscard]] std::shared_ptr<DeviceVsmProductOwner>
prepare_warm_owner(VirtualPipelineState &, const VirtualRunProjection &,
                   const node::accel::detail::DeviceVsmIdentity &,
                   const VirtualDeviceVsmRouteProof &, Status &,
                   const char *&) noexcept;
[[nodiscard]] Status
rearm_prepared(VirtualExecutionDeviceVsmPrepared &) noexcept;
[[nodiscard]] Status
discard_prepared(VirtualPipelineState &,
                 VirtualRunResources &,
                 VirtualExecutionDeviceVsmPrepared &, bool pooled) noexcept;
[[nodiscard]] std::shared_ptr<DeviceVsmProductOwner>
execution_owner(VirtualPipelineState &,
                const VirtualExecutionDeviceVsmPrepared &) noexcept;
void classify_device_vsm_backing_evidence(const DeviceVsmProductOwner &,
                                          DeviceVsmProductEvidence &) noexcept;
[[nodiscard]] bool
reset_device_vsm_run_evidence(DeviceVsmProductOwner &) noexcept;
[[nodiscard]] Status
reserve_device_vsm_capacity(VirtualPipelineState &,
                            storage::Reservation &) noexcept;
[[nodiscard]] Status commit_device_vsm_memory(DeviceVsmProductOwner &) noexcept;
[[nodiscard]] Status prepare_physical_buffers(VirtualPipelineState &,
                                              const VirtualRunProjection &,
                                              const AccelDeviceState &,
                                              DeviceVsmProductOwner &,
                                              const VirtualDeviceVsmRouteProof &,
                                              const char *&) noexcept;
[[nodiscard]] bool
resident_bindings_match(const VirtualPipelineState &,
                        const VirtualRunProjection &,
                        const DeviceVsmProductOwner &) noexcept;
[[nodiscard]] bool graph_resident_bindings_match(
    const VirtualPipelineState &, const VirtualRunProjection &,
    const AccelDeviceState &, const DeviceVsmProductOwner &,
    const node::accel::detail::DeviceVsmProof *) noexcept;
[[nodiscard]] Status
snapshot_pipelines(std::span<const std::shared_ptr<PipelineState>>,
                   std::span<const std::uint32_t>, bool graph,
                   DeviceVsmPipelineSnapshot &) noexcept;
[[nodiscard]] bool project_graph_resident(
    const residency::TiledGraphPlan &, const residency::Pool &,
    std::span<const std::uint32_t> graph_input_resources,
    const node::accel::detail::DeviceVsmGraphWavefrontProof &,
    node::accel::detail::DeviceVsmGraphResidentProof &, const char *&) noexcept;

[[nodiscard]] Status stage_input(DeviceVsmProductRun &) noexcept;
[[nodiscard]] Status stage_output(DeviceVsmProductRun &) noexcept;

[[nodiscard]] Status begin_pipelines(DeviceVsmProductRun &) noexcept;
void reject_pipelines(DeviceVsmProductRun &, Status) noexcept;
[[nodiscard]] bool prepare_success_publication(DeviceVsmProductRun &,
                                               DeviceVsmPublication &) noexcept;
[[nodiscard]] bool prepare_failure_publication(DeviceVsmProductRun &,
                                               DeviceVsmPublication &,
                                               Status) noexcept;
void commit_publication(void *, bool) noexcept;

void complete(void *, node::accel::detail::DeviceVsmFinal &&) noexcept;
[[nodiscard]] node::accel::detail::DeviceVsmFinal
unknown_final(const DeviceVsmProductRun &) noexcept;
[[nodiscard]] Status submit(DeviceVsmProductRun &) noexcept;
[[nodiscard]] VirtualExecutionResult finish(DeviceVsmProductRun &,
                                            Status) noexcept;
void fold_stats(DeviceVsmProductRun &) noexcept;

} // namespace rund::compute::detail::device_vsm_product_detail
