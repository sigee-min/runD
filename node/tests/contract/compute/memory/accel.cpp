#include "../../../../src/compute/buffer/state.hpp"
#include "model.hpp"

#include <accel/check.hpp>
#include <accel/kernel/evidence.hpp>

#include "../../target/selection.hpp"

#include <node/accel/buffer.hpp>
#include <node/runtime/compute/access.hpp>

#include "../../../../src/accel/context/transfer.hpp"
#include "../../../../src/compute/buffer/local.hpp"
#include "../../../../src/compute/device/state.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <unistd.h>

namespace rund_node_memory_contract {

int CheckAccelMemory(const rund::compute::Backend backend) {
  const std::array<std::uint32_t, 4u> values{1u, 2u, 3u, 4u};
  auto device =
      rund::compute::open(rund::node::test_contract::target_for(backend));
  if (!device) {
    std::fprintf(stderr, "memory backend=%u open reason=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(device.error().size()),
                 device.error().data());
    return 1;
  }
  {
    auto buffer = device->upload<std::uint32_t>(values);
    if (!buffer) {
      return 2;
    }
    const std::shared_ptr<rund::compute::detail::BufferState> owner =
        rund::compute::detail::BufferAccess::state(*buffer);
    if (owner == nullptr) {
      return 3;
    }
    const auto projected = rund::compute::detail::planned_buffer_storage_bytes(
        *owner->device, owner->bytes);
    if (!projected || owner->physical_bytes != *projected) {
      return 3;
    }
    const auto active = device->memory();
    std::array<rund::compute::MemoryEntry, 7u> entries{};
    const auto snapshot = device->memory_snapshot(entries);
    bool found_logical = false;
    bool found_committed = false;
    for (std::size_t index = 0u; index < snapshot.written; ++index) {
      const rund::compute::MemoryEntry &entry = entries[index];
      found_logical =
          found_logical ||
          (entry.category == rund::compute::MemoryCategory::Resident &&
           entry.use == rund::compute::MemoryUse::Internal &&
           entry.bytes.current == values.size() * sizeof(std::uint32_t));
      found_committed =
          found_committed ||
          (entry.category == rund::compute::MemoryCategory::Device &&
           entry.use == rund::compute::MemoryUse::Internal &&
           entry.bytes.current == active.device.current);
    }
    if (!ValidStats(active) ||
        active.resident.current != values.size() * sizeof(std::uint32_t) ||
        active.resident.peak != active.resident.current ||
        active.resident.cumulative != active.resident.current ||
        active.device.current != owner->physical_bytes ||
        active.device.peak < active.device.current ||
        active.device.cumulative < active.device.current ||
        active.device.budget < active.device.current ||
        active.transfer.current != 0u || active.transfer.reused != 0u ||
        active.transfer.budget != 0u ||
        active.transfer.cumulative != values.size() * sizeof(std::uint32_t) ||
        active.transfer.peak != values.size() * sizeof(std::uint32_t) ||
        snapshot.truncated() ||
        snapshot.summary.resident.current != active.resident.current ||
        snapshot.summary.device.current != active.device.current ||
        !found_logical || !found_committed) {
      return 3;
    }
  }
  const auto released = device->memory();
  if (!ValidStats(released) || released.resident.current != 0u ||
      released.resident.peak != values.size() * sizeof(std::uint32_t) ||
      released.resident.cumulative != released.resident.peak ||
      released.device.current != 0u || released.device.peak == 0u) {
    return 4;
  }
  if (backend == rund::compute::Backend::Metal) {
    const long host_page = ::sysconf(_SC_PAGESIZE);
    if (host_page <= 4) {
      return 10;
    }
    const auto page = static_cast<std::size_t>(host_page);
    const std::array<std::size_t, 9u> boundary_bytes{4u,
                                                     12u,
                                                     16u,
                                                     page - 4u,
                                                     page,
                                                     page + 4u,
                                                     page * 4u,
                                                     page * 4u + 4u,
                                                     page * 5u + 4u};
    for (const std::size_t bytes : boundary_bytes) {
      auto boundary = device->buffer<std::uint32_t>(bytes / 4u);
      const std::shared_ptr<rund::compute::detail::BufferState> state =
          boundary ? rund::compute::detail::BufferAccess::state(*boundary)
                   : nullptr;
      if (!boundary || state == nullptr) {
        return 10;
      }
      const auto committed =
          rund::compute::detail::planned_buffer_storage_bytes(*state->device,
                                                              state->bytes);
      if (!committed || state->physical_bytes != *committed) {
        return 10;
      }
    }
  }
  {
    using namespace rund::compute;
    auto source = device->upload<std::uint32_t>(values);
    auto target = device->buffer<std::uint32_t>(values.size());
    auto program =
        on(*device)
            .map<std::uint32_t>("memory-admission-no-allocation", values.size(),
                                [](auto value) { return value + 1u; })
            .compile();
    if (!source || !target || !program) {
      return 11;
    }
    auto builder =
        pipeline(*device).then(*program, read(*source), write(*target));
    const auto plan = builder.plan();
    const std::shared_ptr<detail::DeviceState> &state =
        detail::DeviceAccess::state(*device);
    const detail::AccelDeviceState *const accel =
        state == nullptr ? nullptr : detail::accel_device(*state);
    if (!plan || accel == nullptr || plan->committed_peak_bytes == 0u) {
      return 11;
    }
    const std::uint64_t before =
        rund::node::accel::ReadRuntimeStats(accel->pick)
            .run.allocations.buffer_allocation_count;
    auto rejected =
        std::move(builder)
            .budget(MemoryBudget{.bytes = plan->committed_peak_bytes - 1u})
            .prepare();
    const std::uint64_t after = rund::node::accel::ReadRuntimeStats(accel->pick)
                                    .run.allocations.buffer_allocation_count;
    if (rejected || rejected.reason() != Reason::PipelineMemoryBudget ||
        after != before) {
      return 11;
    }
  }
  if (backend == rund::compute::Backend::Vulkan) {
    constexpr std::size_t staging_budget = 1024u * 1024u;
    constexpr std::size_t pool_limit = staging_budget * 32u;
    constexpr std::size_t count = staging_budget / sizeof(std::uint32_t) + 17u;
    std::vector<std::uint32_t> input(count);
    for (std::size_t index = 0u; index < input.size(); ++index) {
      input[index] = static_cast<std::uint32_t>(index * 17u + 3u);
    }
    auto buffer = device->upload<std::uint32_t>(input);
    const std::shared_ptr<rund::compute::detail::DeviceState> &device_state =
        rund::compute::detail::DeviceAccess::state(*device);
    rund::compute::detail::AccelDeviceState *const accel =
        device_state == nullptr
            ? nullptr
            : rund::compute::detail::accel_device(*device_state);
    const std::shared_ptr<rund::compute::detail::BufferState> buffer_state =
        buffer ? rund::compute::detail::BufferAccess::state(*buffer) : nullptr;
    rund::compute::detail::AccelBufferState *const native =
        buffer_state == nullptr
            ? nullptr
            : rund::compute::detail::accel_buffer(*buffer_state);
    std::vector<std::uint32_t> output(count);
    if (accel == nullptr || native == nullptr) {
      return 5;
    }
    const rund::node::accel::detail::AccelTransfer transfer =
        rund::node::accel::detail::DownloadAccelBufferMeasured(
            accel->context, native->buffer, output.data(),
            output.size() * sizeof(std::uint32_t), 0u, false);
    const auto memory = device->memory();
    if (!transfer.check.ok || output != input ||
        transfer.staging_peak_bytes == 0u ||
        transfer.staging_peak_bytes > staging_budget ||
        transfer.command_submits < 2u || memory.staging.current > pool_limit ||
        memory.staging.peak > pool_limit + staging_budget) {
      return 6;
    }
  }
  return 0;
}

} // namespace rund_node_memory_contract
