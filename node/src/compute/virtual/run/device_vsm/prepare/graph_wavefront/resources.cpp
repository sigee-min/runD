#include "internal.hpp"

#include "../../../../../../compute/device/residency/pool.hpp"
#include "../../../../../fixed/format.hpp"

#include <algorithm>
#include <limits>

namespace rund::compute::detail::device_vsm_product_detail {

bool project_graph_resident_resources(GraphResidentDraft &draft) noexcept {
  namespace accel = node::accel::detail;
  for (std::size_t index = 0u; index < draft.resources.size(); ++index) {
    const residency::TiledGraphResource &resource = draft.resources[index];
    if (resource.resource == 0u || resource.physical_id == 0u ||
        resource.page_bytes == 0u || resource.logical_bytes == 0u ||
        resource.type != draft.root_type ||
        resource.page_bytes % draft.result.type.element_bytes != 0u ||
        resource.logical_bytes % draft.result.type.element_bytes != 0u ||
        resource.page_bytes / draft.result.type.element_bytes >
            std::numeric_limits<std::uint32_t>::max() ||
        resource.logical_bytes / draft.result.type.element_bytes >
            std::numeric_limits<std::uint32_t>::max() ||
        !rund::kernel::ComputeFixedFormatAbsent(
            rund::compute::detail::kernel_format(resource.format)) ||
        resource.first_stage >= draft.stages.size() ||
        resource.last_stage >= draft.stages.size()) {
      remember_graph_resident_reason(GraphResidentShapeInvalid, draft.reason);
      return false;
    }
    const bool external_input =
        resource.kind == residency::GraphResourceKind::ExternalInput;
    const bool external_output =
        resource.kind == residency::GraphResourceKind::ExternalOutput;
    if (external_input || external_output) {
      if (resource.persistence != residency::ResourcePersistence::Backing ||
          (external_input &&
           resource.role != residency::GraphResourceRole::Input) ||
          (external_output &&
           resource.role != residency::GraphResourceRole::Output)) {
        remember_graph_resident_reason(GraphResidentShapeInvalid, draft.reason);
        return false;
      }
    } else if (resource.kind != residency::GraphResourceKind::Internal ||
               resource.role != residency::GraphResourceRole::Intermediate ||
               resource.persistence !=
                   residency::ResourcePersistence::Transient) {
      remember_graph_resident_reason(GraphResidentShapeInvalid, draft.reason);
      return false;
    }
    if (!external_input && !external_output) {
      const residency::PoolPhysicalOwner *const physical =
          draft.pool.graph_owner(resource.physical_id);
      if (physical == nullptr || resource.type != draft.root_type ||
          physical->physical_class.type != resource.type ||
          physical->physical_class.format != resource.format ||
          physical->physical_class.page_bytes != resource.page_bytes ||
          physical->color != resource.color ||
          physical->physical_class.role != residency::FrameRole::Intermediate) {
        remember_graph_resident_reason(GraphResidentResourceOwnerInvalid,
                                       draft.reason);
        return false;
      }
    }
    std::size_t owner_slot = draft.internal_count;
    if (!external_input && !external_output) {
      for (std::size_t owner = 0u; owner < draft.internal_count; ++owner) {
        if (draft.internal_owners[owner]->physical_id == resource.physical_id &&
            draft.internal_owners[owner]->color == resource.color) {
          owner_slot = owner;
          break;
        }
      }
      if (owner_slot == draft.internal_count) {
        remember_graph_resident_reason(GraphResidentResourceOwnerInvalid,
                                       draft.reason);
        return false;
      }
    }
    auto &row = draft.result.resources[index];
    row.resource = resource.resource;
    row.physical_id = resource.physical_id;
    row.color = resource.color;
    row.producer = resource.producer_stage;
    row.first = resource.first_stage;
    row.last = resource.last_stage;
    row.owner_slot = external_input || external_output
                         ? accel::DeviceVsmGraphResidentExternalSlotInvalid
                         : static_cast<std::uint32_t>(owner_slot);
    row.external_slot = accel::DeviceVsmGraphResidentExternalSlotInvalid;
    if (external_input) {
      const auto found =
          std::find(draft.graph_input_resources.begin(),
                    draft.graph_input_resources.end(), resource.resource);
      if (found == draft.graph_input_resources.end()) {
        remember_graph_resident_reason(GraphResidentShapeInvalid, draft.reason);
        return false;
      }
      row.external_slot = static_cast<std::uint32_t>(
          std::distance(draft.graph_input_resources.begin(), found));
    } else if (external_output) {
      if (++draft.output_count != 1u) {
        remember_graph_resident_reason(GraphResidentShapeInvalid, draft.reason);
        return false;
      }
      row.external_slot =
          static_cast<std::uint32_t>(draft.graph_input_resources.size());
    }
    row.page_bytes = resource.page_bytes;
    row.logical_bytes = resource.logical_bytes;
    row.role = static_cast<std::uint8_t>(resource.role);
    row.type = static_cast<std::uint8_t>(resource.type);
    row.format = rund::compute::detail::kernel_format(resource.format);
    row.valid = 1u;
  }
  if (draft.output_count != 1u) {
    remember_graph_resident_reason(GraphResidentShapeInvalid, draft.reason);
    return false;
  }
  return true;
}

} // namespace rund::compute::detail::device_vsm_product_detail
