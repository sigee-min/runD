#include "proof.hpp"

#include <kernel/core/checked.hpp>

namespace rund::node::accel::detail {

std::uint64_t
device_vsm_output_page_count(const DeviceVsmProof &proof) noexcept {
  return proof.topology == DeviceVsmTopology::GraphMapReduce ||
                 proof.topology == DeviceVsmTopology::Reduce
             ? 1u
             : proof.geometry.page_count;
}

bool device_vsm_backing_valid(const rund::kernel::ResidentBufferRef &backing,
                              const std::shared_ptr<void> &handle,
                              const std::uint32_t usage,
                              const std::uint32_t element_bytes,
                              const std::uint64_t required_bytes) noexcept {
  if (backing.id == 0u || backing.bytes == 0u || handle == nullptr ||
      backing.usage != usage || backing.element_bytes != element_bytes ||
      backing.stride_bytes != element_bytes || required_bytes == 0u ||
      required_bytes % element_bytes != 0u ||
      backing.count < required_bytes / element_bytes ||
      backing.offset_bytes > backing.bytes ||
      required_bytes > backing.bytes - backing.offset_bytes) {
    return false;
  }
  return true;
}

bool device_vsm_residents_valid(const DeviceVsmProof &proof) noexcept {
  const DeviceVsmResidentSet &residents = proof.residents;
  std::uint64_t total_input_bytes = 0u;
  if (residents.input_count == 0u || residents.output_count == 0u ||
      residents.count != residents.input_count + residents.output_count ||
      residents.count > residents.rows.size() ||
      residents.input_count != proof.plan.input_buffer_count ||
      residents.output_count != proof.plan.output_buffer_count ||
      !::rund::kernel::checked::mul(proof.geometry.logical_bytes,
                                   residents.input_count, total_input_bytes) ||
      total_input_bytes == 0u) {
    return false;
  }
  for (std::size_t index = 0u; index < residents.count; ++index) {
    const bool input = index < residents.input_count;
    const DeviceVsmResidentBinding &row = residents.rows[index];
    const std::uint32_t usage =
        input ? (proof.topology == DeviceVsmTopology::Window &&
                         proof.window.mutates_input
                     ? rund::kernel::kResidentUsageWrite
                     : rund::kernel::kResidentUsageRead)
              : rund::kernel::kResidentUsageWrite;
    const std::uint64_t required_bytes =
        input ? proof.geometry.logical_bytes : proof.output_bytes;
    if (row.role != (input ? DeviceVsmResidentRole::Input
                           : DeviceVsmResidentRole::Output) ||
        !device_vsm_backing_valid(row.backing, row.handle, usage,
                                  proof.geometry.element_bytes,
                                  required_bytes)) {
      return false;
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (residents.rows[prior].backing.id == row.backing.id ||
          residents.rows[prior].handle == row.handle) {
        return false;
      }
    }
  }
  for (std::size_t index = residents.count; index < residents.rows.size();
       ++index) {
    if (residents.rows[index].backing.id != 0u ||
        residents.rows[index].handle != nullptr) {
      return false;
    }
  }
  return true;
}

bool device_vsm_proof_valid(const DeviceVsmProof &proof) noexcept {
  if (!proof.identity || proof.semantic_owner == nullptr ||
      proof.artifact == nullptr || !proof.artifact->ok || !proof.plan.ok ||
      proof.plan.api == rund::kernel::ComputeApi::Cpu ||
      proof.artifact->key.api != proof.plan.api ||
      proof.artifact->key.variant !=
          rund::kernel::LoweringArtifactVariant::DeviceVsm ||
      !device_vsm_runtime_geometry_valid(proof.geometry) ||
      !device_vsm_topology_valid(proof) || proof.width < 2u ||
      proof.width > 4u || proof.width > proof.geometry.page_count ||
      !proof.fixed_common_storage || proof.windows == nullptr ||
      proof.window_count == 0u ||
      proof.plan.dispatch_count != proof.window_count ||
      proof.parameter_bytes != proof.plan.param_bytes ||
      (proof.parameter_bytes != 0u && proof.parameters == nullptr) ||
      !device_vsm_residents_valid(proof)) {
    return false;
  }
  for (std::uint64_t index = 0u; index < proof.window_count; ++index) {
    if (proof.windows[index].tile_count == 0u) {
      return false;
    }
  }
  DeviceVsmPageProjection first{};
  DeviceVsmPageProjection last{};
  return device_vsm_project_page(proof.geometry, 0u, first) &&
         device_vsm_project_page(proof.geometry,
                                 proof.geometry.page_count - 1u, last) &&
         first.core_offset == 0u &&
         last.core_offset < proof.geometry.logical_bytes &&
         last.core_bytes == proof.geometry.logical_bytes - last.core_offset;
}

} // namespace rund::node::accel::detail
