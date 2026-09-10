#include "internal.hpp"

#include <algorithm>
#include <limits>

namespace rund::node::accel::detail::device_vsm_graph_resident {

bool validate_resources(const DeviceVsmGraphResidentProof &proof,
                        const std::uint32_t external_input_count,
                        ValidationContext &context) noexcept {
  for (std::size_t index = 0u; index < proof.resource_count; ++index) {
    const auto &resource = proof.resources[index];
    if (resource.valid == 0u || resource.resource == 0u ||
        resource.physical_id == 0u || resource.first > resource.last ||
        resource.last >= proof.stage_count ||
        (resource.producer != std::numeric_limits<std::uint32_t>::max() &&
         resource.producer >= proof.stage_count) ||
        resource.page_bytes == 0u ||
        resource.page_bytes % proof.type.element_bytes != 0u ||
        resource.page_bytes / proof.type.element_bytes !=
            context.payload_elements ||
        resource.logical_bytes == 0u ||
        resource.logical_bytes % proof.type.element_bytes != 0u ||
        resource.logical_bytes / proof.type.element_bytes >
            std::numeric_limits<std::uint32_t>::max() ||
        resource.type != device_vsm_graph_resident_type_code(proof.type) ||
        !rund::kernel::ComputeFixedFormatAbsent(resource.format)) {
      return false;
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (proof.resources[prior].resource == resource.resource) {
        return false;
      }
    }
    if (resource.role == 1u) {
      if (resource.owner_slot >= proof.owner_binding_count ||
          resource.external_slot != DeviceVsmGraphResidentExternalSlotInvalid) {
        return false;
      }
      context.owner_used[resource.owner_slot] = true;
      const auto &owner = proof.owners[resource.owner_slot];
      if (owner.physical_id != resource.physical_id ||
          owner.color != resource.color ||
          owner.physical_page_bytes != resource.page_bytes ||
          owner.physical_format != resource.format ||
          owner.physical_type != resource.type ||
          owner.physical_type != device_vsm_graph_resident_type_code(proof.type) ||
          owner.physical_role != resource.role) {
        return false;
      }
    } else if (resource.role == 0u || resource.role == 2u) {
      if (resource.owner_slot != DeviceVsmGraphResidentExternalSlotInvalid ||
          resource.external_slot == DeviceVsmGraphResidentExternalSlotInvalid) {
        return false;
      }
    } else {
      return false;
    }
  }
  if (external_input_count != std::numeric_limits<std::uint32_t>::max()) {
    if (external_input_count == 0u ||
        external_input_count >= DeviceVsmGraphResidentResourceCapacity) {
      return false;
    }
    std::array<bool, DeviceVsmGraphResidentResourceCapacity> slots{};
    std::size_t input_count = 0u;
    std::size_t output_count = 0u;
    for (std::size_t index = 0u; index < proof.resource_count; ++index) {
      const auto &resource = proof.resources[index];
      if (resource.role == 0u) {
        if (resource.external_slot >= external_input_count ||
            slots[resource.external_slot]) {
          return false;
        }
        slots[resource.external_slot] = true;
        ++input_count;
      } else if (resource.role == 2u) {
        if (resource.external_slot != external_input_count ||
            ++output_count != 1u) {
          return false;
        }
      }
    }
    if (input_count != external_input_count || output_count != 1u ||
        !std::all_of(slots.begin(), slots.begin() + external_input_count,
                     [](const bool value) { return value; })) {
      return false;
    }
    if (device_vsm_page_map_active(proof.page_map)) {
      if (proof.page_map.words[3u] == 0u ||
          proof.page_map.words[3u] > external_input_count) {
        return false;
      }
      for (std::size_t index = 0u; index < proof.page_map.words[3u]; ++index) {
        DeviceVsmPageMapRow map_row{};
        if (!device_vsm_page_map_load_row(proof.page_map, index, map_row)) {
          return false;
        }
        const DeviceVsmGraphResidentResource *resource = nullptr;
        for (std::size_t resource_index = 0u;
             resource_index < proof.resource_count; ++resource_index) {
          const auto &candidate = proof.resources[resource_index];
          if (candidate.role == 0u &&
              candidate.external_slot == map_row.external_slot) {
            resource = &candidate;
            break;
          }
        }
        if (resource == nullptr || resource->resource != map_row.resource ||
            resource->page_bytes != map_row.page_bytes) {
          return false;
        }
      }
    }
  }
  return true;
}

} // namespace rund::node::accel::detail::device_vsm_graph_resident
