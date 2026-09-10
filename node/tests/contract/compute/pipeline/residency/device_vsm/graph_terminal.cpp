#include "local.hpp"

#include "src/accel/kernel/residency/device_vsm/terminal.hpp"

#include <memory>
#include <string_view>

namespace rund_node_test_pipeline_residency::device_vsm_test {

bool CheckGraphTerminal() {
  auto proof = std::make_shared<accel::DeviceVsmProof>();
  proof->identity = {.hi = 7u, .lo = 8u};
  proof->topology = accel::DeviceVsmTopology::GraphMapReduce;
  proof->geometry.page_count = 5u;
  proof->geometry.logical_bytes = 5u * 16u * sizeof(std::uint64_t) - 24u;
  proof->output_bytes = sizeof(std::uint64_t);
  proof->graph_map_reduce.semantic.op = rund::kernel::ReduceOp::Max;
  proof->graph_map_reduce.wavefront = accel::DeviceVsmGraphWavefrontProof{
      .same_dispatch = {0u, 1u},
      .prior_release = {2u, 0u},
      .map_stage = 0u,
      .collective_stage = 1u,
      .stage_count = 2u,
      .frame_capacity = 2u,
      .batch_count = 3u,
  };
  std::uint32_t steps = 0u;
  std::uint32_t trace = 0u;
  if (!accel::device_vsm_graph_wavefront_expected(
          proof->graph_map_reduce.wavefront, steps, trace)) {
    return false;
  }
  accel::DeviceVsmSubmissionControl submission_control{};
  const accel::DeviceVsmRequest request{
      .proof = proof,
      .token = 9u,
      .generation = 10u,
      .nonce = 11u,
      .submission_control = &submission_control,
  };
  if (!submission_control.accept()) {
    return false;
  }
  accel::DeviceVsmNativeExecution native{
      .accepted = {true, "ok"},
      .counters = {5u, 5u, 5u, 5u, 1u, 1u, 0u, accel::DeviceVsmNoFailedPage,
                   steps, trace},
      .result_acquired = true,
  };
  const accel::DeviceVsmFinal exact =
      accel::ClassifyDeviceVsmTerminal(request, native);
  if (!exact.check.ok || exact.terminal != accel::DeviceVsmTerminal::Known ||
      exact.evidence.graph_wavefront_steps != steps ||
      exact.evidence.graph_wavefront_trace != trace) {
    return false;
  }
  native.counters[accel::DeviceVsmGraphWavefrontTraceWord] ^= 1u;
  const accel::DeviceVsmFinal mismatched =
      accel::ClassifyDeviceVsmTerminal(request, native);
  return !mismatched.check.ok &&
         mismatched.terminal == accel::DeviceVsmTerminal::UnknownMayWrite &&
         std::string_view{mismatched.check.reason} == "compute_device_lost";
}

} // namespace rund_node_test_pipeline_residency::device_vsm_test
