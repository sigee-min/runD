#include "../../../../accel/kernel/prepared/interface/api.hpp"
#include "../pipeline.hpp"

#include "../../../../accel/context/internal/support.hpp"
#include "../../../../accel/context/local.hpp"
#include "../../../job/state.hpp"
#include "../../../pipeline/state.hpp"
#include "../../../status.hpp"

#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>

#include <memory>
#include <utility>

namespace rund::compute::detail::accel_backend {

MemoryCounter device_staging(const DeviceState &device) noexcept {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return {};
  }
  const node::accel::AccelMemoryCounter memory =
      node::accel::ReadAccelMemoryStats(accel->pick).staging;
  return MemoryCounter{.current = memory.current,
                       .peak = memory.peak,
                       .cumulative = memory.cumulative,
                       .reused = memory.reused,
                       .budget = memory.budget};
}

MemoryCounter job_staging(const JobState &job) noexcept {
  node::accel::detail::PreparedMemory memory =
      node::accel::detail::ReadPreparedKernelMemory(job.prepared);
  const node::accel::detail::PreparedMemory pending =
      node::accel::detail::ReadPreparedKernelMemory(job.write_prepared);
  node::accel::detail::accumulate_serial_memory(memory, pending);
  return MemoryCounter{.current = memory.current,
                       .peak = memory.peak,
                       .cumulative = memory.cumulative,
                       .reused = memory.reused,
                       .budget = memory.budget};
}

node::accel::detail::PreparedPipelineMemory
pipeline_memory(const PipelineState &pipeline) noexcept {
  node::accel::detail::PreparedPipelineMemory memory =
      node::accel::detail::ReadPreparedKernelPipelineMemory(pipeline.prepared);
  const node::accel::detail::PreparedPipelineMemory alternate =
      node::accel::detail::ReadPreparedKernelPipelineMemory(
          pipeline.alternate_prepared);
  node::accel::detail::accumulate_serial_memory(memory.host, alternate.host);
  node::accel::detail::accumulate_serial_memory(memory.device,
                                                alternate.device);
  node::accel::detail::accumulate_serial_memory(memory.staging,
                                                alternate.staging);
  // Primary and transactional alternate share one immutable template
  // registry. Report that owner once after combining the two stream-local
  // pipelines so retained host memory is neither omitted nor doubled.
  node::accel::detail::accumulate_serial_memory(
      memory.host,
      node::accel::detail::ReadPreparedKernelTemplateRegistryMemory(
          pipeline.accel_templates));
  return memory;
}

} // namespace rund::compute::detail::accel_backend
