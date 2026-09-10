#include "internal.hpp"

namespace rund::node::accel::detail::device_vsm_graph_resident {

bool validate_tails(const DeviceVsmGraphResidentProof &proof) noexcept {
  for (std::size_t index = proof.stage_count;
       index < DeviceVsmGraphResidentStageCapacity; ++index) {
    if (proof.stages[index].valid != 0u ||
        proof.stages[index].port_count != 0u) {
      return false;
    }
  }
  for (std::size_t index = proof.resource_count;
       index < DeviceVsmGraphResidentResourceCapacity; ++index) {
    const auto &resource = proof.resources[index];
    if (resource.valid != 0u || resource.resource != 0u ||
        resource.physical_id != 0u || resource.color != 0u ||
        resource.producer != 0u || resource.first != 0u ||
        resource.last != 0u || resource.owner_slot != 0u ||
        resource.external_slot != 0u || resource.page_bytes != 0u ||
        resource.logical_bytes != 0u || resource.role != 0u ||
        resource.type != 0u || resource.format.integer_bits != 0u ||
        resource.format.fraction_bits != 0u) {
      return false;
    }
  }
  for (std::size_t index = proof.owner_binding_count;
       index < DeviceVsmGraphResidentPhysicalCapacity; ++index) {
    const auto &owner = proof.owners[index];
    if (owner.valid != 0u || owner.physical_id != 0u || owner.color != 0u ||
        owner.view != 0u || owner.bank_count != 0u ||
        owner.physical_page_bytes != 0u || owner.physical_type != 0u ||
        owner.physical_tier != 0u || owner.physical_role != 0u ||
        owner.physical_format.integer_bits != 0u ||
        owner.physical_format.fraction_bits != 0u) {
      return false;
    }
    for (std::size_t bank = 0u; bank < DeviceVsmGraphResidentBankCapacity;
         ++bank) {
      if (owner.refs[bank].id != 0u || owner.refs[bank].bytes != 0u ||
          owner.refs[bank].offset_bytes != 0u ||
          owner.refs[bank].element_bytes != 0u ||
          owner.refs[bank].stride_bytes != 0u ||
          owner.refs[bank].count != 0u || owner.refs[bank].usage != 0u ||
          owner.handles[bank] != nullptr || owner.local_first[bank] != 0u ||
          owner.cache_regions[bank].first != 0u ||
          owner.cache_regions[bank].count != 0u ||
          owner.bank_regions[bank].first != 0u ||
          owner.bank_regions[bank].count != 0u ||
          owner.buffer_generations[bank] != 0u) {
        return false;
      }
    }
  }
  return true;
}

} // namespace rund::node::accel::detail::device_vsm_graph_resident
