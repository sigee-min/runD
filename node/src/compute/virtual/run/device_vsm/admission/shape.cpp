#include "shape.hpp"

#include "../../../../type.hpp"

#include <kernel/program/compute/reduce/model.hpp>
#include <kernel/program/compute/scan/model.hpp>

namespace rund::compute::detail::device_vsm_product_detail::admission_detail {
namespace {

[[nodiscard]] bool pointwise_shape(const VirtualPipelineState &state,
                                   const VirtualRunProjection &run) noexcept {
  return (state.geometry.route == VirtualRoute::Pointwise ||
          state.geometry.route == VirtualRoute::MultiPointwise) &&
         !run.clamp_window && !run.clip_window &&
         run.input_page_bytes == run.input_payload_bytes &&
         run.output_page_bytes == run.output_payload_bytes &&
         run.input_prefix_bytes == 0u && run.output_prefix_bytes == 0u;
}

[[nodiscard]] bool
multi_pointwise_shape(const VirtualPipelineState &state,
                      const VirtualRunProjection &run) noexcept {
  return state.geometry.route == VirtualRoute::MultiPointwise &&
         run.multi_pointwise() && run.input_count == state.input_count &&
         run.input_count >= 2u &&
         run.input_count <= VirtualPipelineState::InputCapacity;
}

[[nodiscard]] bool multi_scan_shape(const VirtualPipelineState &state,
                                    const VirtualRunProjection &run) noexcept {
  return state.geometry.route == VirtualRoute::Scan && run.scan() &&
         run.multi_scan() && run.input_count == state.input_count &&
         run.input_count >= 2u &&
         run.input_count <= VirtualPipelineState::InputCapacity;
}

[[nodiscard]] bool window_shape(const VirtualPipelineState &state,
                                const VirtualRunProjection &run) noexcept {
  return state.geometry.route == VirtualRoute::Window &&
         (run.clamp_window || run.clip_window) && run.input_type == Type::U32 &&
         run.output_type == Type::U32 &&
         run.input_page_bytes == run.output_page_bytes &&
         run.input_prefix_bytes == run.output_prefix_bytes &&
         run.input_prefix_bytes != 0u &&
         run.input_prefix_bytes <= run.input_page_bytes &&
         run.input_payload_bytes <=
             run.input_page_bytes - run.input_prefix_bytes &&
         run.input_prefix_bytes == run.input_page_bytes -
                                       run.input_prefix_bytes -
                                       run.input_payload_bytes;
}

[[nodiscard]] std::size_t
graph_stage_count(const VirtualPipelineState &state) noexcept {
  return state.pipeline == nullptr || state.pipeline->residency == nullptr
             ? 0u
             : state.pipeline->residency->tiled_graph().stages().size();
}

[[nodiscard]] bool graph_map_reduce_shape(const VirtualPipelineState &state,
                                          const VirtualRunProjection &run,
                                          const std::size_t stages) noexcept {
  return state.geometry.route == VirtualRoute::GraphReduction &&
         run.graph_reduction() && run.reduction() && !run.scan() &&
         run.input_type == Type::U64 && run.output_type == Type::U64 &&
         run.active.output_bytes == sizeof(std::uint64_t) &&
         run.output_page_bytes == sizeof(std::uint64_t) &&
         run.output_payload_bytes == sizeof(std::uint64_t) &&
         run.input_count == state.input_count &&
         run.input_count == state.graph_input_resource_count &&
         run.input_count != 0u &&
         run.input_count < node::accel::detail::DeviceVsmResidentCapacity &&
         (run.operation == static_cast<std::uint32_t>(kernel::ReduceOp::Sum) ||
          run.operation ==
              static_cast<std::uint32_t>(kernel::ReduceOp::CountNonzero) ||
          run.operation == static_cast<std::uint32_t>(kernel::ReduceOp::Min) ||
          run.operation == static_cast<std::uint32_t>(kernel::ReduceOp::Max)) &&
         state.pipeline != nullptr && state.pipeline->residency != nullptr &&
         state.alternate_pipeline != nullptr &&
         graph_terminal_pipeline(state, 0u) != nullptr &&
         graph_terminal_pipeline(state, 1u) != nullptr && stages >= 2u &&
         stages <= residency::execution::UseCapacity &&
         state.graph_pipelines.size() == stages * residency::Pool::BankCount &&
         ((stages == 2u && state.device_vsm_semantic_pipeline == nullptr) ||
          (stages > 2u && state.device_vsm_semantic_pipeline != nullptr));
}

[[nodiscard]] bool graph_pointwise_shape(const VirtualPipelineState &state,
                                         const VirtualRunProjection &run,
                                         const std::size_t stages) noexcept {
  return state.geometry.route == VirtualRoute::GraphPointwise &&
         run.graph_execution() && !run.graph_reduction() && !run.reduction() &&
         !run.scan() &&
         (run.input_type == Type::U32 || run.input_type == Type::U64) &&
         run.output_type == run.input_type && run.input_count != 0u &&
         run.input_count == state.input_count &&
         run.input_count == state.graph_input_resource_count &&
         run.input_count < node::accel::detail::DeviceVsmResidentCapacity &&
         run.active.input_bytes == run.active.output_bytes &&
         run.input_page_bytes == run.output_page_bytes &&
         run.input_payload_bytes == run.output_payload_bytes &&
         state.pipeline != nullptr && state.pipeline->residency != nullptr &&
         graph_terminal_pipeline(state, 0u) != nullptr && stages >= 2u &&
         stages <= node::accel::detail::DeviceVsmGraphStageCapacity &&
         state.graph_pipelines.size() == stages * residency::Pool::BankCount;
}

[[nodiscard]] bool scan_shape(const VirtualPipelineState &state,
                              const VirtualRunProjection &run,
                              const bool multi_scan) noexcept {
  return state.geometry.route == VirtualRoute::Scan && run.scan() &&
         !run.reduction() && !run.graph_reduction() && !run.clamp_window &&
         !run.clip_window &&
         (run.input_type == Type::U32 || run.input_type == Type::U64) &&
         (!run.multi_scan() || run.input_type == Type::U64) &&
         run.output_type == run.input_type &&
         run.input_count == state.input_count &&
         ((run.multi_scan() && multi_scan) ||
          (!run.multi_scan() && run.input_count == 1u)) &&
         (run.operation ==
              static_cast<std::uint32_t>(kernel::ScanOp::InclusiveSum) ||
          run.operation ==
              static_cast<std::uint32_t>(kernel::ScanOp::ExclusiveSum)) &&
         run.input_page_bytes == run.output_page_bytes &&
         run.input_payload_bytes == run.output_payload_bytes &&
         run.input_prefix_bytes == run.output_prefix_bytes;
}

[[nodiscard]] bool reduce_shape(const VirtualPipelineState &state,
                                const VirtualRunProjection &run,
                                const std::size_t element_bytes) noexcept {
  return state.geometry.route == VirtualRoute::Reduction && run.reduction() &&
         !run.graph_reduction() && !run.scan() && !run.clamp_window &&
         !run.clip_window &&
         (run.input_type == Type::U32 || run.input_type == Type::U64) &&
         run.output_type == run.input_type &&
         (run.operation == static_cast<std::uint32_t>(kernel::ReduceOp::Sum) ||
          run.operation ==
              static_cast<std::uint32_t>(kernel::ReduceOp::CountNonzero) ||
          run.operation == static_cast<std::uint32_t>(kernel::ReduceOp::Min) ||
          run.operation == static_cast<std::uint32_t>(kernel::ReduceOp::Max)) &&
         run.output_page_bytes == element_bytes &&
         run.output_payload_bytes == element_bytes &&
         run.active.output_bytes == element_bytes &&
         run.input_page_bytes == run.input_payload_bytes &&
         run.input_prefix_bytes == 0u && run.output_prefix_bytes == 0u;
}

} // namespace

Shape classify(const VirtualPipelineState &state,
               const VirtualRunProjection &run,
               const std::size_t element_bytes) noexcept {
  const bool multi_scan = multi_scan_shape(state, run);
  const std::size_t stages = graph_stage_count(state);
  return Shape{
      .pointwise = pointwise_shape(state, run),
      .multi_pointwise = multi_pointwise_shape(state, run),
      .multi_scan = multi_scan,
      .window = window_shape(state, run),
      .stages = stages,
      .graph_map_reduce = graph_map_reduce_shape(state, run, stages),
      .graph_pointwise = graph_pointwise_shape(state, run, stages),
      .scan = scan_shape(state, run, multi_scan),
      .reduce = reduce_shape(state, run, element_bytes),
  };
}

} // namespace
  // rund::compute::detail::device_vsm_product_detail::admission_detail
