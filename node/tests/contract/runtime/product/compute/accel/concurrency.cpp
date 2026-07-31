#include "local.hpp"

#include "../../support.hpp"
#include "src/accel/kernel/prepared.hpp"
#include "src/accel/vulkan/command/ring.hpp"
#include "src/compute/device/state.hpp"
#include "src/compute/job/state.hpp"

#include <node/accel/buffer.hpp>
#include <node/runtime/compute/access.hpp>
#include <rund/compute.hpp>
#include <rund/compute/session.hpp>

#include <atomic>
#include <cstdio>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace rund::node::test_contract {

namespace {

struct VulkanCapacityGate final {
  std::atomic_bool release{false};
  std::atomic_bool ok{true};
  std::atomic_size_t completed{};
};

void CompleteVulkanCapacity(void *const raw,
                            const rund::AccelEvidence &evidence) noexcept {
  auto *const gate = static_cast<VulkanCapacityGate *>(raw);
  if (gate == nullptr) {
    return;
  }
  if (!evidence.ok) {
    gate->ok.store(false, std::memory_order_release);
  }
  gate->release.wait(false, std::memory_order_acquire);
  gate->completed.fetch_add(1u, std::memory_order_release);
  gate->completed.notify_all();
}

void ReleaseVulkanCapacity(VulkanCapacityGate &gate,
                           const std::size_t expected) noexcept {
  gate.release.store(true, std::memory_order_release);
  gate.release.notify_all();
  while (gate.completed.load(std::memory_order_acquire) != expected) {
    const std::size_t observed = gate.completed.load(std::memory_order_relaxed);
    gate.completed.wait(observed, std::memory_order_acquire);
  }
}

} // namespace

int CheckVulkanCommandCapacity(
    compute::Program<std::int32_t(std::int32_t)> &program,
    const std::span<const std::int32_t> input) {
  constexpr std::size_t capacity =
      ::rund::node::accel::detail::kVulkanCommandCapacity;
  std::vector<compute::Job<std::int32_t(std::int32_t)>> jobs{};
  jobs.reserve(capacity + 1u);
  for (std::size_t index = 0u; index <= capacity; ++index) {
    auto job = program.resident(input);
    if (!job) {
      return 1;
    }
    jobs.push_back(std::move(*job));
  }

  const auto first = compute::detail::JobAccess::state(jobs.front());
  if (first == nullptr || first->program == nullptr ||
      first->program->device == nullptr) {
    return 2;
  }
  auto *const device = compute::detail::accel_device(*first->program->device);
  if (device == nullptr) {
    return 3;
  }

  ::rund::node::accel::ResetRuntimeStats(device->pick);
  VulkanCapacityGate gate{};
  std::size_t submitted_count = 0u;
  for (std::size_t index = 0u; index < capacity; ++index) {
    const auto state = compute::detail::JobAccess::state(jobs[index]);
    const rund::AccelCheck submitted =
        node::accel::detail::SubmitPreparedKernel(
            device->context, state->prepared, state, CompleteVulkanCapacity,
            &gate);
    if (!submitted.ok) {
      ReleaseVulkanCapacity(gate, submitted_count);
      return 4;
    }
    ++submitted_count;
  }

  const auto overflow = compute::detail::JobAccess::state(jobs.back());
  const rund::AccelCheck rejected = node::accel::detail::SubmitPreparedKernel(
      device->context, overflow->prepared, overflow, CompleteVulkanCapacity,
      &gate);
  const rund::RuntimeStats pressure =
      ::rund::node::accel::ReadRuntimeStats(device->pick);
  const bool bounded =
      !rejected.ok &&
      std::string_view{rejected.reason} == "accel_vulkan_command_capacity" &&
      pressure.ok && pressure.command_capacity == capacity &&
      pressure.command_inflight_peak == capacity &&
      pressure.command_capacity_rejection_count == 1u &&
      pressure.buffer_allocation_count == 0u &&
      pressure.pipeline_compile_count == 0u;

  ReleaseVulkanCapacity(gate, capacity);
  if (!bounded || !gate.ok.load(std::memory_order_acquire)) {
    return 5;
  }

  std::uint64_t graph_hash = 0u;
  std::uint64_t output_hash = 0u;
  for (auto &job : jobs) {
    if (!job.run()) {
      return 6;
    }
    const auto output = job.read();
    if (!output ||
        *output != std::vector<std::int32_t>{-1, 5, 13, 23, 29, -11, 19, 47}) {
      return 7;
    }
    const compute::Stats stats = job.stats();
    // One compute submission plus one explicit host readback submission.
    // Stats reports physical queue work, so the transfer is not hidden.
    if (stats.command_submits != 2u || stats.dispatches == 0u ||
        stats.command_capacity != capacity ||
        stats.command_inflight_peak == 0u ||
        stats.command_inflight_peak > stats.command_capacity ||
        stats.command_capacity_rejections != 0u) {
      return 8;
    }
    if (graph_hash == 0u) {
      graph_hash = stats.graph_hash;
      output_hash = stats.output_hash;
    } else if (stats.graph_hash != graph_hash ||
               stats.output_hash != output_hash) {
      return 9;
    }
  }
  if (graph_hash == 0u || output_hash == 0u) {
    return 10;
  }
  const rund::RuntimeStats retained =
      ::rund::node::accel::ReadRuntimeStats(device->pick);
  const std::uint64_t minimum_submits = capacity + jobs.size();
  return retained.ok && retained.command_capacity_rejection_count == 1u &&
                 retained.command_submit_count >= minimum_submits
             ? 0
             : 11;
}

int CheckComputeAccelConcurrency(
    ::rund::Session &session, const compute::Target target,
    compute::Program<std::int32_t(std::int32_t)> &program,
    const std::span<const std::int32_t> first,
    const std::span<const std::int32_t> second) {
  auto first_job = program.resident(first);
  auto second_job = program.resident(second);
  if (!first_job || !second_job) {
    return 1;
  }
  auto first_task = session.compute(*first_job).submit();
  auto second_task = session.compute(*second_job).submit();
  const auto first_result = first_task.wait();
  const auto second_result = second_task.wait();
  if (!first_result || !second_result) {
    const std::string_view first_reason =
        first_result ? std::string_view{"ok"} : first_result.error();
    const std::string_view second_reason =
        second_result ? std::string_view{"ok"} : second_result.error();
    std::fprintf(stderr, "concurrent accel first=%.*s second=%.*s\n",
                 static_cast<int>(first_reason.size()), first_reason.data(),
                 static_cast<int>(second_reason.size()), second_reason.data());
    return 2;
  }
  const compute::Stats first_stats = first_result.stats();
  const compute::Stats second_stats = second_result.stats();
  if (first_stats.backend != target.backend() ||
      second_stats.backend != target.backend() ||
      first_stats.command_submits != 1u || second_stats.command_submits != 1u ||
      first_stats.dispatches == 0u || second_stats.dispatches == 0u) {
    return 3;
  }
  if (target.backend() == compute::Backend::Vulkan &&
      (first_stats.command_capacity == 0u ||
       first_stats.command_capacity != second_stats.command_capacity ||
       first_stats.command_inflight_peak == 0u ||
       first_stats.command_inflight_peak > first_stats.command_capacity ||
       second_stats.command_inflight_peak == 0u ||
       second_stats.command_inflight_peak > second_stats.command_capacity ||
       first_stats.command_capacity_rejections != 0u ||
       second_stats.command_capacity_rejections != 0u)) {
    return 4;
  }
  const auto first_output = first_job->read();
  const auto second_output = second_job->read();
  if (!first_output || !second_output ||
      *first_output !=
          std::vector<std::int32_t>{-1, 5, 13, 23, 29, -11, 19, 47} ||
      *second_output !=
          std::vector<std::int32_t>{21, 19, 17, 15, 13, 11, 9, 7}) {
    return 5;
  }

  return CheckComputeAccelScanConcurrency(session, target);
}

} // namespace rund::node::test_contract
