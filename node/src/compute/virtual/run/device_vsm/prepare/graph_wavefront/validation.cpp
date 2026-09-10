#include "internal.hpp"

#include "../../../../../../accel/kernel/residency/device_vsm/graph_wavefront.hpp"

#include <algorithm>
#include <limits>

namespace rund::compute::detail::device_vsm_product_detail {

const char GraphResidentShapeInvalid[] =
    "device_vsm_graph_resident_shape_invalid";
const char GraphResidentResourceOwnerInvalid[] =
    "device_vsm_graph_resident_resource_owner_invalid";
const char GraphResidentOwnerArenaInvalid[] =
    "device_vsm_graph_resident_owner_arena_invalid";
const char GraphResidentBankBindingInvalid[] =
    "device_vsm_graph_resident_bank_binding_invalid";
const char GraphResidentStagePortsInvalid[] =
    "device_vsm_graph_resident_stage_ports_invalid";
const char GraphResidentProofValidationInvalid[] =
    "device_vsm_graph_resident_proof_validation_invalid";

void remember_graph_resident_reason(const char *const key,
                                    const char *&reason) noexcept {
  if (reason == nullptr) {
    reason = key;
  }
}

bool validate_graph_resident_shape(GraphResidentDraft &draft) noexcept {
  namespace accel = node::accel::detail;
  const auto resources = draft.resources;
  const auto stages = draft.stages;
  const auto owners = draft.owners;
  if (stages.size() < 2u ||
      stages.size() > accel::DeviceVsmGraphResidentStageCapacity ||
      resources.size() < 2u ||
      resources.size() > accel::DeviceVsmGraphResidentResourceCapacity ||
      owners.empty() ||
      owners.size() > accel::DeviceVsmGraphResidentPhysicalCapacity ||
      draft.graph_input_resources.empty() ||
      draft.graph_input_resources.size() >=
          accel::DeviceVsmGraphResidentResourceCapacity) {
    remember_graph_resident_reason(GraphResidentShapeInvalid, draft.reason);
    return false;
  }
  draft.root_type = resources.front().type;
  const TypeProjection root_projection = project_type(draft.root_type);
  if ((draft.root_type != Type::U32 && draft.root_type != Type::U64) ||
      root_projection.bytes == 0u ||
      root_projection.bytes > std::numeric_limits<std::uint32_t>::max() ||
      !accel::device_vsm_graph_resident_type_valid(
          accel::DeviceVsmGraphResidentType{
              .scalar = root_projection.scalar,
              .domain = root_projection.domain,
              .element_bytes =
                  static_cast<std::uint32_t>(root_projection.bytes)})) {
    remember_graph_resident_reason(GraphResidentShapeInvalid, draft.reason);
    return false;
  }
  draft.result.type = accel::DeviceVsmGraphResidentType{
      .scalar = root_projection.scalar,
      .domain = root_projection.domain,
      .element_bytes = static_cast<std::uint32_t>(root_projection.bytes)};
  if (!accel::device_vsm_graph_wavefront_valid(draft.wavefront,
                                               draft.plan.page_count())) {
    remember_graph_resident_reason(GraphResidentProofValidationInvalid,
                                   draft.reason);
    return false;
  }
  if (draft.wavefront.stage_count != stages.size() ||
      draft.wavefront.frame_capacity != draft.plan.frame_capacity()) {
    remember_graph_resident_reason(GraphResidentShapeInvalid, draft.reason);
    return false;
  }
  for (std::size_t input = 0u; input < draft.graph_input_resources.size();
       ++input) {
    const std::uint32_t resource_id = draft.graph_input_resources[input];
    if (resource_id == 0u ||
        std::find(draft.graph_input_resources.begin(),
                  draft.graph_input_resources.begin() +
                      static_cast<std::ptrdiff_t>(input),
                  resource_id) != draft.graph_input_resources.begin() +
                                      static_cast<std::ptrdiff_t>(input)) {
      remember_graph_resident_reason(GraphResidentShapeInvalid, draft.reason);
      return false;
    }
    const residency::TiledGraphResource *const resource =
        draft.plan.resource(resource_id);
    if (resource == nullptr ||
        resource->kind != residency::GraphResourceKind::ExternalInput ||
        resource->role != residency::GraphResourceRole::Input ||
        resource->persistence != residency::ResourcePersistence::Backing) {
      remember_graph_resident_reason(GraphResidentShapeInvalid, draft.reason);
      return false;
    }
  }
  for (const residency::TiledGraphPhysicalClass &owner : owners) {
    if (owner.role == residency::GraphResourceRole::Intermediate) {
      if (draft.internal_count >= draft.internal_owners.size()) {
        remember_graph_resident_reason(GraphResidentShapeInvalid, draft.reason);
        return false;
      }
      draft.internal_owners[draft.internal_count++] = &owner;
    }
  }
  if (draft.internal_count == 0u) {
    remember_graph_resident_reason(GraphResidentShapeInvalid, draft.reason);
    return false;
  }
  return true;
}

} // namespace rund::compute::detail::device_vsm_product_detail
