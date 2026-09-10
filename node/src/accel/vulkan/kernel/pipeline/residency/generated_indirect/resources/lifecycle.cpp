#include "../internal.hpp"
#include "../map.hpp"
#include "../map_state.hpp"

#include "../../../../../map/api.hpp"
#include "../../../../../map/admission.hpp"
#include "../../../../../map/encode/control.hpp"
#include "../../../../../map/source/control.hpp"
#include "../../../../../map/local.hpp"
#include "../../../prepare/record.hpp"
#include "../../sliding/internal.hpp"

#include <cstdint>

namespace rund::node::accel::detail::vulkan_generated_indirect_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck prepare(VulkanPipeline &pipeline,
                         VulkanResidencySelection &selection) noexcept {
  if (selection.mode == VulkanResidencyMode::GraphStageGeneratedIndirect) {
    return impl::prepare_graph(pipeline, selection);
  }
  if (selection.mode == VulkanResidencyMode::GraphStageSequence) {
    return impl::prepare_sequence(pipeline, selection);
  }
  return {false, "accel_kernel_pipeline_invalid"};
}

void destroy(VulkanResidencySelection &selection) noexcept {
  if (selection.mode == VulkanResidencyMode::GraphStageGeneratedIndirect) {
    impl::destroy_graph(selection);
  } else if (selection.mode == VulkanResidencyMode::GraphStageSequence) {
    impl::destroy_sequence(selection);
  }
}

void destroy_roles(VulkanResidencySelection &selection) noexcept {
  if (selection.adapter == nullptr) {
    return;
  }
  for (auto &role : selection.sliding.generated_roles) {
    for (VulkanCommand &command : role) {
      DestroyCommand(selection.adapter->device, command);
    }
  }
  for (VulkanCommand &command : selection.sliding.generated_tail) {
    DestroyCommand(selection.adapter->device, command);
  }
}

#endif

} // namespace rund::node::accel::detail::vulkan_generated_indirect_detail
