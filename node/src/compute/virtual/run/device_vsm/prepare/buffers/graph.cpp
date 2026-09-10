#include "local.hpp"

#include "../../endpoint_mode.hpp"

#include "../../../../../../accel/context/local.hpp"
#include "../../../../../../accel/context/transfer.hpp"
#include "../../../../../type.hpp"

namespace rund::compute::detail::device_vsm_product_detail {
namespace {

namespace accel = node::accel::detail;

[[nodiscard]] bool
same_ref(const rund::kernel::ResidentBufferRef &left,
         const rund::kernel::ResidentBufferRef &right) noexcept {
  return left.id == right.id && left.bytes == right.bytes &&
         left.offset_bytes == right.offset_bytes &&
         left.element_bytes == right.element_bytes &&
         left.stride_bytes == right.stride_bytes && left.count == right.count &&
         left.usage == right.usage;
}

[[nodiscard]] bool
same_physical_ref(const rund::kernel::ResidentBufferRef &left,
                  const rund::kernel::ResidentBufferRef &right) noexcept {
  return left.id == right.id && left.bytes == right.bytes &&
         left.offset_bytes == right.offset_bytes &&
         left.element_bytes == right.element_bytes &&
         left.stride_bytes == right.stride_bytes && left.count == right.count;
}

} // namespace

bool graph_resident_bindings_match(
    const VirtualPipelineState &state, const VirtualRunProjection &run,
    const AccelDeviceState &native, const DeviceVsmProductOwner &owner,
    const node::accel::detail::DeviceVsmProof *const expected) noexcept {
  if (run.input_count == 0u || run.input_count != owner.input_count ||
      run.input_count != state.input_count || state.output == nullptr ||
      state.output->backing == nullptr) {
    return false;
  }
  if (owner.route_proof.kind() != VirtualDeviceVsmRouteKind::GraphResident ||
      (owner.route_proof.endpoint() != VirtualDeviceVsmEndpoint::Resident &&
       owner.route_proof.endpoint() != VirtualDeviceVsmEndpoint::Staged)) {
    return false;
  }
  EndpointSet endpoints{};
  const EndpointMode observed = classify_endpoints(state, run, endpoints);
  const EndpointMode sealed =
      owner.route_proof.endpoint() == VirtualDeviceVsmEndpoint::Resident
          ? EndpointMode::Resident
          : EndpointMode::Staged;
  if (observed != sealed) {
    return false;
  }
  const bool all_resident = sealed == EndpointMode::Resident;
  const bool all_staged = sealed == EndpointMode::Staged;
  if ((all_resident && owner.resident_output != endpoints.output) ||
      (all_staged &&
       (owner.resident_output != nullptr ||
        owner.output_staging.size() != run.active.output_bytes)) ||
      (all_resident &&
       (!same_resident(state.output, owner.resident_output) ||
        VirtualBackingAccess::resident(*state.output->backing) == nullptr)) ||
      (all_staged &&
       VirtualBackingAccess::resident(*state.output->backing) != nullptr)) {
    return false;
  }
  if (expected != nullptr &&
      (expected->residents.input_count != owner.input_count ||
       expected->residents.output_count != 1u ||
       expected->residents.count != owner.input_count + 1u)) {
    return false;
  }
  const auto check = [&](const std::size_t index,
                         const std::shared_ptr<VirtualBufferState> &buffer,
                         const std::shared_ptr<BufferState> &resident,
                         const rund::AccelBuffer &prepared,
                         const bool output) noexcept {
    const std::uint64_t bytes =
        output ? run.active.output_bytes : run.active.input_bytes;
    const std::uint64_t element_bytes =
        type_bytes(output ? run.output_type : run.input_type);
    const bool expected_staged = all_staged;
    const bool has_resident = resident != nullptr;
    const bool staging_exact =
        expected_staged ? (output ? owner.output_staging.size() == bytes
                                  : owner.input_staging[index].size() == bytes)
                        : (output ? owner.output_staging.empty()
                                  : owner.input_staging[index].empty());
    if (buffer == nullptr || buffer->backing == nullptr ||
        prepared.handle == nullptr || !staging_exact ||
        (has_resident == expected_staged) ||
        (expected_staged &&
         VirtualBackingAccess::resident(*buffer->backing) != nullptr) ||
        (!expected_staged &&
         (!same_resident(buffer, resident) ||
          VirtualBackingAccess::resident(*buffer->backing) == nullptr))) {
      return false;
    }
    const accel::UploadRoute current =
        accel::ProjectAccelBufferRoute(native.context, prepared);
    if (current.handle == nullptr || current.resident.id == 0u || bytes == 0u ||
        element_bytes == 0u || current.bytes < bytes ||
        current.resident.bytes < bytes || current.resident.offset_bytes != 0u ||
        current.resident.element_bytes != element_bytes ||
        current.resident.stride_bytes != element_bytes ||
        bytes % element_bytes != 0u ||
        current.resident.count < bytes / element_bytes) {
      return false;
    }
    if (!expected_staged) {
      const AccelBufferState *const storage = accel_buffer(*resident);
      if (storage == nullptr || !storage->buffer) {
        return false;
      }
      const accel::UploadRoute admitted =
          accel::ProjectAccelBufferRoute(native.context, storage->buffer);
      if (admitted.handle == nullptr ||
          !same_physical_ref(admitted.resident, current.resident) ||
          !accel::device_vsm_graph_resident_same_object(admitted.handle,
                                                        current.handle)) {
        return false;
      }
    }
    if (expected == nullptr) {
      return true;
    }
    const auto &row = expected->residents.rows[index];
    return row.role == (output ? accel::DeviceVsmResidentRole::Output
                               : accel::DeviceVsmResidentRole::Input) &&
           same_ref(current.resident, row.backing) &&
           accel::device_vsm_graph_resident_same_object(current.handle,
                                                        row.handle);
  };
  for (std::size_t index = 0u; index < owner.input_count; ++index) {
    if (!check(index, state.inputs[index], owner.resident_inputs[index],
               owner.inputs[index], false)) {
      return false;
    }
  }
  return check(owner.input_count, state.output, owner.resident_output,
               owner.output, true);
}

} // namespace rund::compute::detail::device_vsm_product_detail
