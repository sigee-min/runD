#include "local.hpp"

#include "../../endpoint_mode.hpp"

#include "../../../../../../accel/context/local.hpp"
#include "../../../../../../accel/context/transfer.hpp"

namespace rund::compute::detail::device_vsm_product_detail {
namespace {

namespace accel = node::accel::detail;

[[nodiscard]] bool
staged_views_match(const VirtualPipelineState &state,
                   const VirtualRunProjection &run,
                   const DeviceVsmProductOwner &owner) noexcept {
  const AccelDeviceState *const native =
      state.pipeline == nullptr || state.pipeline->device == nullptr
          ? nullptr
          : accel_device(*state.pipeline->device);
  if (native == nullptr ||
      owner.route_proof.kind == VirtualDeviceVsmRouteKind::GraphResident ||
      owner.route_proof.endpoint != VirtualDeviceVsmEndpoint::Staged ||
      owner.input_count != 1u || owner.resident_output != nullptr ||
      owner.resident_inputs[0u] != nullptr) {
    return false;
  }
  const auto input = accel::WriteAccelBuffer(native->context, owner.inputs[0u]);
  const auto output = accel::ReadAccelBuffer(native->context, owner.output);
  return input && input.bytes >= run.active.input_bytes && output &&
         output.bytes >= run.active.output_bytes;
}

} // namespace

bool same_resident(const std::shared_ptr<VirtualBufferState> &buffer,
                   const std::shared_ptr<BufferState> &resident) noexcept {
  return buffer != nullptr && buffer->backing != nullptr &&
         VirtualBackingAccess::resident(*buffer->backing) == resident;
}

bool resident_bindings_match(const VirtualPipelineState &state,
                             const VirtualRunProjection &run,
                             const DeviceVsmProductOwner &owner) noexcept {
  if (run.input_count != owner.input_count ||
      run.input_count != state.input_count || state.output == nullptr ||
      state.output->backing == nullptr) {
    return false;
  }
  const bool graph_resident =
      owner.route_proof.kind == VirtualDeviceVsmRouteKind::GraphResident;
  const bool staged =
      owner.route_proof.endpoint == VirtualDeviceVsmEndpoint::Staged;
  const bool mapped = !graph_resident && staged;
  const bool all_resident =
      graph_resident
          ? owner.route_proof.endpoint == VirtualDeviceVsmEndpoint::Resident
          : owner.resident_output != nullptr;
  const bool all_staged =
      owner.resident_output == nullptr &&
      (mapped ? owner.output_staging.empty()
              : owner.output_staging.size() == run.active.output_bytes);
  if (!all_resident && !all_staged) {
    return false;
  }
  if (all_resident &&
      (!same_resident(state.output, owner.resident_output) ||
       !VirtualBackingAccess::resident(*state.output->backing))) {
    return false;
  }
  if (all_staged &&
      VirtualBackingAccess::resident(*state.output->backing) != nullptr) {
    return false;
  }
  if (all_staged && mapped && !staged_views_match(state, run, owner)) {
    return false;
  }
  for (std::size_t index = 0u; index < owner.input_count; ++index) {
    if (state.inputs[index] == nullptr ||
        state.inputs[index]->backing == nullptr) {
      return false;
    }
    if (all_resident &&
        (owner.resident_inputs[index] == nullptr ||
         !same_resident(state.inputs[index], owner.resident_inputs[index]) ||
         VirtualBackingAccess::resident(*state.inputs[index]->backing) ==
             nullptr ||
         !owner.input_staging[index].empty())) {
      return false;
    }
    if (all_staged && (owner.resident_inputs[index] != nullptr ||
                       (mapped ? !owner.input_staging[index].empty()
                               : owner.input_staging[index].size() !=
                                     run.active.input_bytes) ||
                       VirtualBackingAccess::resident(
                           *state.inputs[index]->backing) != nullptr)) {
      return false;
    }
  }
  return true;
}

} // namespace rund::compute::detail::device_vsm_product_detail
