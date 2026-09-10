#include "../internal.hpp"

#include "../operations.hpp"

namespace rund::compute::detail::device_vsm_product_detail {

std::shared_ptr<DeviceVsmProductOwner>
execution_owner(VirtualPipelineState &state,
                const VirtualExecutionDeviceVsmPrepared &prepared) noexcept {
  const auto owner =
      std::static_pointer_cast<DeviceVsmProductOwner>(prepared.owner);
  const bool graph =
      owner != nullptr && owner->proof != nullptr &&
      (owner->proof->topology ==
           node::accel::detail::DeviceVsmTopology::GraphMapReduce ||
       owner->proof->topology ==
           node::accel::detail::DeviceVsmTopology::GraphPointwise ||
       owner->proof->topology ==
           node::accel::detail::DeviceVsmTopology::GraphResident);
  const VirtualRunTopology topology =
      !graph ? VirtualRunTopology::Direct
      : owner->proof->topology ==
              node::accel::detail::DeviceVsmTopology::GraphMapReduce
          ? VirtualRunTopology::GraphReduction
          : VirtualRunTopology::GraphPointwise;
  std::array<std::shared_ptr<PipelineState>, DeviceVsmPipelineCapacity>
      pipelines{};
  std::array<std::uint32_t, DeviceVsmPipelineCapacity> stages{};
  std::size_t count = 0u;
  const bool selected =
      select_pipelines(state, topology, pipelines, stages, count);
  bool same = selected && owner != nullptr && owner->pipeline_count == count;
  for (std::size_t index = 0u; same && index < count; ++index) {
    same = owner->pipelines[index] == pipelines[index] &&
           owner->pipeline_stages[index] == stages[index];
  }
  return !prepared || owner == nullptr || owner->proof == nullptr ||
                 !owner->preparation || owner->registration == nullptr ||
                 state.device_vsm_product_cache != prepared.owner || !same ||
                 owner->input_count != owner->proof->residents.input_count ||
                 prepared.proof.page_count() !=
                     owner->proof->geometry.page_count ||
                 owner->proof->output_bytes == 0u
             ? std::shared_ptr<DeviceVsmProductOwner>{}
             : owner;
}

} // namespace rund::compute::detail::device_vsm_product_detail
