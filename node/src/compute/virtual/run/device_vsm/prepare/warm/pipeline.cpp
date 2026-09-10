#include "local.hpp"

namespace rund::compute::detail::device_vsm_product_detail::warm_detail {

bool pipelines_match(VirtualPipelineState &state,
                     const VirtualRunProjection &run,
                     const DeviceVsmProductOwner &owner) noexcept {
  std::array<std::shared_ptr<PipelineState>, DeviceVsmPipelineCapacity>
      pipelines{};
  std::array<std::uint32_t, DeviceVsmPipelineCapacity> stages{};
  std::size_t count = 0u;
  if (!select_pipelines(state, run, pipelines, stages, count) ||
      owner.pipeline_count != count) {
    return false;
  }
  for (std::size_t index = 0u; index < count; ++index) {
    if (owner.pipelines[index] != pipelines[index] ||
        owner.pipeline_stages[index] != stages[index]) {
      return false;
    }
  }
  return true;
}

bool backend_bindings_match(const VirtualPipelineState &state,
                            const VirtualRunProjection &run,
                            const AccelDeviceState *const native,
                            const DeviceVsmProductOwner &owner) noexcept {
  if (owner.proof == nullptr ||
      owner.proof->topology !=
          node::accel::detail::DeviceVsmTopology::GraphResident) {
    return true;
  }
  return native != nullptr &&
         graph_resident_bindings_match(state, run, *native, owner,
                                       owner.proof.get());
}

} // namespace rund::compute::detail::device_vsm_product_detail::warm_detail
