#pragma once

#include <accel/check.hpp>

#include "../../command/run.hpp"
#include "encode.hpp"
#include "input.hpp"
#include "output.hpp"
#include "readback.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
bool ExecuteWindow(MetalAdapter &adapter,
                   const std::shared_ptr<void> &pipeline_handle,
                   const rund::kernel::ComputePlan &plan,
                   const rund::kernel::ComputeDispatchWindow &window,
                   const rund::kernel::BindingSet &bindings,
                   const MetalRuntimeBuffer *const param_buffer) {
  id<MTLComputePipelineState> pipeline =
      (__bridge id<MTLComputePipelineState>)pipeline_handle.get();
  if (pipeline == nil || param_buffer == nullptr ||
      param_buffer->buffer == nullptr) {
    return false;
  }

  ScopedMetalBuffers scoped{adapter};
  const StagedProof staged = PlanStagedWindow(bindings, window);
  if (!staged.ok()) {
    return false;
  }
  MetalRuntimeBuffer *input = nullptr;
  if (!PrepareStagedInputBuffer(adapter, plan, window, bindings, staged, scoped,
                                input)) {
    return false;
  }
  MetalRuntimeBuffer *output = nullptr;
  std::size_t output_size = 0u;
  if (!PrepareStagedOutputBuffer(adapter, plan, window, scoped, output,
                                 output_size)) {
    return false;
  }
  if (output == nullptr) {
    return false;
  }
  CommandRun command{};
  if (!OpenCommand(adapter, command).ok) {
    return false;
  }
  const bool encoded = EncodeWindow(command.encoder, pipeline, plan, window,
                                    bindings, *param_buffer, input, *output);
  const rund::AccelCheck submit = FinishCommand(
      adapter, command,
      rund::AccelCheck{encoded, encoded ? "ok" : "compute_binding_mismatch"});
  if (!submit.ok) {
    return false;
  }
  if (!ScatterStagedOutput(adapter, window, bindings, staged, *output,
                           output_size)) {
    return false;
  }
  RecordMetalDispatch(adapter);
  return true;
}
#endif

} // namespace rund::node::accel::detail
