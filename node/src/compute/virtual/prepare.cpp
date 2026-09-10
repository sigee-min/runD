#include "local.hpp"

#include "../backend.hpp"
#include "../device/residency/pool.hpp"
#include "../graph/compile/slice.hpp"
#include "../graph/compile/slice/semantic.hpp"
#include "../pipeline/residency/integration.hpp"
#include "../pipeline/residency/planner.hpp"
#include "backing.hpp"
#include "host_ring.hpp"
#include "prepare/internal.hpp"
#include "prepare/residency/local.hpp"
#include "stats.hpp"

#include <kernel/core/checked.hpp>
#include <rund/compute/pipeline/builder.hpp>
#include <rund/compute/pipeline/capacity.hpp>
#include <rund/compute/pipeline/runtime.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <utility>

namespace rund::compute::detail {

Result<std::shared_ptr<VirtualPipelineState>> prepare_virtual_pipeline(
    const std::shared_ptr<ProgramState> &program,
    const std::span<const std::shared_ptr<VirtualBufferState>> inputs,
    const std::shared_ptr<VirtualBufferState> &output,
    const ResidencyConfig config) noexcept {
  return prepare_virtual_pipeline(program, inputs, output, config,
                                  GraphPageMap{});
}

Result<std::shared_ptr<VirtualPipelineState>> prepare_virtual_pipeline(
    const std::shared_ptr<ProgramState> &program,
    const std::span<const std::shared_ptr<VirtualBufferState>> inputs,
    const std::shared_ptr<VirtualBufferState> &output,
    const ResidencyConfig config, const GraphPageMap page_map) noexcept {
  if (program == nullptr || inputs.empty() || output == nullptr ||
      inputs.size() > VirtualPipelineState::InputCapacity) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        Reason::PipelineInvalid);
  }
  const std::shared_ptr<VirtualBufferState> &input = inputs.front();
  if (input == nullptr || output->backing == nullptr) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        Reason::PipelineInvalid);
  }
  for (std::size_t index = 0u; index < inputs.size(); ++index) {
    if (inputs[index] == nullptr || inputs[index] == output ||
        inputs[index]->backing == nullptr ||
        inputs[index]->backing == output->backing) {
      return Result<std::shared_ptr<VirtualPipelineState>>::fail(
          Reason::PipelineInvalid);
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (inputs[index] == inputs[prior] ||
          inputs[index]->backing == inputs[prior]->backing) {
        return Result<std::shared_ptr<VirtualPipelineState>>::fail(
            Reason::PipelineInvalid);
      }
    }
  }
  const Status valid =
      validate_virtual_pipeline_program(*program, inputs, *output);
  if (!valid) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(valid.reason());
  }
  if (output->backing->size_bytes() != output->bytes ||
      std::any_of(inputs.begin(), inputs.end(), [](const auto &candidate) {
        return candidate->backing->size_bytes() != candidate->bytes;
      })) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        Reason::PipelineCapacity);
  }
  const Status capability =
      virtual_prepare_detail::validate_pipeline_capability(*program->device);
  if (!capability) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        capability.reason());
  }

  auto geometry = virtual_prepare_detail::classify_geometry(*program);
  if (!geometry) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        Reason::PrimitiveUnsupported);
  }
  if (geometry->route == VirtualRoute::GraphReduction) {
    return prepare_virtual_graph_reduction(program, inputs, output, config,
                                           page_map, *geometry);
  }
  enum class FuseState : std::uint8_t { Untried, Ready, Rejected };
  struct FuseProbe {
    FuseState state{FuseState::Untried};
    std::shared_ptr<ProgramState> program;
    Reason reason{Reason::PipelineInvalid};
  } fuse;
  if (geometry->route == VirtualRoute::Pointwise &&
      program->graph_info.nodes.size() > 1u) {
    auto compiled = graph_compile::compile_service_free_map_program(program);
    if (compiled) {
      fuse.program = std::move(compiled).value();
      fuse.state = FuseState::Ready;
    } else {
      fuse.reason = compiled.reason();
      fuse.state = FuseState::Rejected;
    }
    if (fuse.state == FuseState::Rejected &&
        (fuse.reason == Reason::PrimitiveUnsupported ||
         fuse.reason == Reason::ExpressionCapacity)) {
      auto sliced =
          graph_compile::compile_tiled_graph_pointwise_slices(program);
      if (sliced) {
        geometry->route = VirtualRoute::GraphPointwise;
        geometry->intermediate_frame_elements = geometry->input_frame_elements;
        geometry->materialization_hi = program->graph_info.fingerprint.hi;
        geometry->materialization_lo = program->graph_info.fingerprint.lo;
        return prepare_virtual_graph_pointwise(program, inputs, output, config,
                                               page_map, *geometry);
      }
    }
  }
  if (!page_map.empty()) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        Reason::PrimitiveUnsupported);
  }
  if (inputs.size() > 1u) {
    return prepare_virtual_multi_device_vsm(program, inputs, output, config,
                                            *geometry);
  }
  return virtual_prepare_detail::prepare_residency_pipeline(
      program, input, output, config, *geometry, fuse.program);
}

} // namespace rund::compute::detail
