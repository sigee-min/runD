#include "../device/state.hpp"
#include "../status.hpp"

#include "../../accel/cpu/caps.hpp"

#if defined(RUND_NODE_OPEN_PROBE)
#include "probe.hpp"
#endif

#include <memory>
#include <thread>
#include <unistd.h>

namespace rund::compute::detail {
namespace {

#if defined(RUND_NODE_OPEN_PROBE)
thread_local std::uint64_t *open_probe_count = nullptr;
#endif

[[nodiscard]] Result<std::shared_ptr<DeviceState>>
open_cpu_config(const std::uint32_t configured_workers) {
#if defined(RUND_NODE_OPEN_PROBE)
  observe_open_config();
#endif
  const kernel::CpuCaps caps = node::accel::detail::DetectCpuCaps();
  if (!caps.ok) {
    return Result<std::shared_ptr<DeviceState>>::fail(
        project_reason(caps.reason, Reason::BackendUnsupported));
  }
  const unsigned hardware = std::thread::hardware_concurrency();
  const std::uint32_t width =
      configured_workers != 0u
          ? configured_workers
          : static_cast<std::uint32_t>(hardware == 0u ? 1u : hardware);
  try {
    node::BackendSelection workers = node::select_backend(width);
    if (!workers) {
      return Result<std::shared_ptr<DeviceState>>::fail(Reason::BackendFailed);
    }
    auto state = std::make_shared<DeviceState>();
    state->backend = Backend::Cpu;
    state->storage = CpuDeviceState{
        .caps = caps,
        .workers = std::move(workers),
    };
    return Result<std::shared_ptr<DeviceState>>::success(std::move(state));
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<DeviceState>>::fail(Reason::DeviceCapacity);
  }
}

} // namespace

#if defined(RUND_NODE_OPEN_PROBE)
ScopedOpenProbe::ScopedOpenProbe(std::uint64_t &count) noexcept
    : previous_(open_probe_count) {
  open_probe_count = &count;
}

ScopedOpenProbe::~ScopedOpenProbe() { open_probe_count = previous_; }

void observe_open_config() noexcept {
  if (open_probe_count != nullptr) {
    ++*open_probe_count;
  }
}
#endif

Result<std::shared_ptr<DeviceState>> open_cpu(const std::uint32_t workers) {
  return open_cpu_config(workers);
}

Status initialize_device_state(
    DeviceState &state,
    const DevicePipelineMemoryLimit pipeline_memory) noexcept {
  const long page_bytes = ::sysconf(_SC_PAGESIZE);
  if (page_bytes <= 0) {
    return Status::fail(Reason::DeviceCapacity);
  }
  storage::Budget budget{pipeline_memory.bytes};
  if (!budget) {
    return Status::fail(Reason::DevicePipelineMemoryCapacity);
  }
  state.host_page_bytes = static_cast<std::uint64_t>(page_bytes);
  state.pipeline_memory_budget = std::move(budget);
  return Status::success();
}

Result<std::shared_ptr<DeviceState>> open_target(const Target target) {
  return open_target(target, DevicePipelineMemoryLimit{});
}

Result<std::shared_ptr<DeviceState>>
open_target(const Target target,
            const DevicePipelineMemoryLimit pipeline_memory) {
  auto device = TargetAccess::open(target)(target.workers());
  if (!device) {
    return device;
  }
  const Status initialized = initialize_device_state(**device, pipeline_memory);
  if (!initialized) {
    return Result<std::shared_ptr<DeviceState>>::fail(initialized.reason());
  }
  return device;
}

Result<std::shared_ptr<DeviceState>> open_target(const Target target,
                                                 const Compile resources) {
  auto device = open_target(target);
  if (!device) {
    return device;
  }
  const Status configured = own_compile(*device, resources);
  return configured
             ? device
             : Result<std::shared_ptr<DeviceState>>::fail(configured.reason());
}

Result<std::shared_ptr<DeviceState>>
open_target(const Target target, const Compile resources,
            const DevicePipelineMemoryLimit pipeline_memory) {
  auto device = open_target(target, pipeline_memory);
  if (!device) {
    return device;
  }
  const Status configured = own_compile(*device, resources);
  return configured
             ? device
             : Result<std::shared_ptr<DeviceState>>::fail(configured.reason());
}

Backend device_backend(const std::shared_ptr<DeviceState> &state) noexcept {
  return state->backend;
}

} // namespace rund::compute::detail
