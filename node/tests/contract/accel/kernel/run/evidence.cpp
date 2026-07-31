#include <accel/check.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run/binding.hpp>
#include <accel/runtime.hpp>

#include "local.hpp"
#include <node/accel/buffer.hpp>
#include <node/accel/context.hpp>

#include <array>
#include <type_traits>

namespace node_accel_contract::kernel_case {

static_assert(std::is_trivially_copyable_v<rund::AccelEvidence>);

[[nodiscard]] bool
ResidentRunEvidenceMatches(const ResidentRunFixture &fixture) {
  const std::array<rund::AccelRunBinding, 2u> bindings = Bindings(fixture);
  const rund::AccelEvidence first_run = rund::node::accel::RunAccelKernel(
      fixture.context, fixture.kernel,
      RunRequest(bindings, fixture.host_input.size(), true));
  const rund::RuntimeStats after_first_run =
      rund::node::accel::ReadRuntimeStats(fixture.context.pick);
  if (!first_run.ok || first_run.graph_id_hi != fixture.kernel.graph_id_hi ||
      first_run.graph_id_lo != fixture.kernel.graph_id_lo ||
      first_run.kernel_id != fixture.kernel.kernel_id ||
      first_run.backend != fixture.context.api ||
      first_run.dispatch_count == 0u ||
      after_first_run.device_to_host_bytes != 0u) {
    return false;
  }

  const rund::AccelEvidence second_run = rund::node::accel::RunAccelKernel(
      fixture.context, fixture.kernel,
      RunRequest(bindings, fixture.host_input.size(), true));
  const rund::RuntimeStats after_second_run =
      rund::node::accel::ReadRuntimeStats(fixture.context.pick);
  const std::uint64_t reuse_count = second_run.pipeline_cache_hit_count +
                                    second_run.buffer_reuse_hit_count +
                                    second_run.descriptor_reuse_hit_count;
  if (!second_run.ok || second_run.dispatch_count == 0u || reuse_count == 0u ||
      after_second_run.device_to_host_bytes != 0u) {
    return false;
  }

  const rund::AccelEvidence cumulative_run = rund::node::accel::RunAccelKernel(
      fixture.context, fixture.kernel,
      RunRequest(bindings, fixture.host_input.size(), false));
  const rund::RuntimeStats after_cumulative_run =
      rund::node::accel::ReadRuntimeStats(fixture.context.pick);
  if (!cumulative_run.ok ||
      cumulative_run.dispatch_count <= second_run.dispatch_count ||
      cumulative_run.device_to_host_bytes != 0u ||
      after_cumulative_run.device_to_host_bytes != 0u) {
    return false;
  }

  std::array<rund::kernel::i32, 8u> downloaded{};
  const rund::AccelCheck download = rund::node::accel::DownloadAccelBuffer(
      fixture.context, fixture.output, downloaded.data(), sizeof(downloaded));
  const rund::RuntimeStats after_download =
      rund::node::accel::ReadRuntimeStats(fixture.context.pick);
  return download.ok && HashFixedLane32(downloaded) == fixture.expected_hash &&
         after_download.device_to_host_bytes >= sizeof(downloaded);
}

} // namespace node_accel_contract::kernel_case
