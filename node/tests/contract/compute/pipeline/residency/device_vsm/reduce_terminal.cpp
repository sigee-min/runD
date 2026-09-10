#include "local.hpp"

#include "src/accel/kernel/residency/device_vsm/terminal.hpp"

#include <memory>
#include <string_view>

namespace rund_node_test_pipeline_residency::device_vsm_test {

bool CheckReduceTerminal() {
  auto proof = std::make_shared<accel::DeviceVsmProof>();
  proof->identity = {.hi = 1u, .lo = 2u};
  proof->topology = accel::DeviceVsmTopology::Reduce;
  proof->geometry.page_count = 5u;
  proof->geometry.logical_bytes = 77u * sizeof(std::uint32_t);
  proof->output_bytes = sizeof(std::uint32_t);
  proof->reduce.semantic.op = rund::kernel::ReduceOp::CountNonzero;
  accel::DeviceVsmSubmissionControl submission_control{};
  const accel::DeviceVsmRequest request{
      .proof = proof,
      .token = 3u,
      .generation = 4u,
      .nonce = 5u,
      .submission_control = &submission_control,
  };
  if (!submission_control.accept()) {
    return false;
  }
  accel::DeviceVsmNativeExecution native{
      .accepted = {true, "ok"},
      .counters = {5u, 5u, 5u, 2u, 0u, 0u, 1u, 2u},
      .result_acquired = true,
  };
  const accel::DeviceVsmFinal count =
      accel::ClassifyDeviceVsmTerminal(request, native);
  if (count.terminal != accel::DeviceVsmTerminal::UnknownMayWrite ||
      count.check.ok ||
      std::string_view{count.check.reason} != "compute_device_lost" ||
      !count.evidence.may_write) {
    return false;
  }
  proof->reduce.semantic.op = rund::kernel::ReduceOp::Sum;
  const accel::DeviceVsmFinal sum =
      accel::ClassifyDeviceVsmTerminal(request, native);
  return sum.terminal == accel::DeviceVsmTerminal::Known && !sum.check.ok &&
         std::string_view{sum.check.reason} == "compute_reduce_sum_overflow" &&
         !sum.evidence.may_write;
}

} // namespace rund_node_test_pipeline_residency::device_vsm_test
