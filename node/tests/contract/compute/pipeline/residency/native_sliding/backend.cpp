#include "local.hpp"

#include <rund/compute/reason.hpp>

#include <algorithm>
#include <chrono>
#include <utility>

namespace rund_node_test_pipeline_residency::native_sliding {

FakeSlidingBackend::~FakeSlidingBackend() {
  {
    std::lock_guard lock{gate};
    stop = true;
    ready.notify_all();
  }
  if (worker.joinable()) {
    worker.join();
  }
}

[[nodiscard]] accel::KernelResult
FakeResult(const std::uint32_t generation, const std::size_t local_count,
           const bool unknown, const bool known_failure,
           const char *const failure_reason) noexcept {
  return accel::KernelResult{
      .check = unknown
                   ? rund::AccelCheck{false, "compute_device_lost"}
                   : (known_failure ? rund::AccelCheck{false, failure_reason}
                                    : rund::AccelCheck{true, "ok"}),
      .stats =
          rund::RuntimeStats{
              .run = {.work = {.dispatch_count = local_count,
                               .command_submit_count = 1u,
                               .command_inflight_peak = 1u}},
              .outcome = {.ok = !unknown,
                          .reason = unknown ? "compute_device_lost" : "ok"},
          },
      .pipeline =
          {
              .control =
                  accel::PreparedPipelineControl{
                      .generation = generation,
                      .reason = static_cast<std::uint32_t>(
                          unknown ? rund::compute::Reason::DeviceLost
                                  : rund::compute::Reason::Ok),
                      .failed_step =
                          unknown ? 0u : accel::PreparedPipelineNoStep,
                      .verified_prefix =
                          static_cast<std::uint32_t>(local_count),
                  },
              .control_command_count = 1u,
              .submitted = true,
              .control_observed = true,
          },
      .terminal = unknown ? accel::NativeTerminal::UnknownMayWrite
                          : accel::NativeTerminal::Known,
  };
}

[[nodiscard]] accel::BackendResidencySlidingCapability
FakeCapability(const std::shared_ptr<void> &raw,
               const accel::ResidencySlidingMemory memory) noexcept {
  const auto backend = std::static_pointer_cast<FakeSlidingBackend>(raw);
  if (backend == nullptr ||
      memory != accel::ResidencySlidingMemory::HostCoherent) {
    return {};
  }
  return accel::BackendResidencySlidingCapability{
      .check = {true, "ok"},
      .memory = accel::ResidencySlidingMemory::HostCoherent,
      .retained_bytes = 0u,
      .transient_bytes = 64u,
      .max_slots = 4u,
      .callbacks_async = true,
      .descriptor_release_acquire = true,
  };
}

[[nodiscard]] accel::BackendResidencySlidingCapability
RejectedCapability(const std::shared_ptr<void> &raw,
                   const accel::ResidencySlidingMemory memory) noexcept {
  accel::BackendResidencySlidingCapability capability =
      FakeCapability(raw, memory);
  capability.descriptor_release_acquire = false;
  return capability;
}

[[nodiscard]] rund::AccelCheck
FakeSeed(const std::shared_ptr<void> &raw,
         const std::uint32_t generation) noexcept {
  const auto backend = std::static_pointer_cast<FakeSlidingBackend>(raw);
  if (backend == nullptr) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  std::unique_lock lock{backend->gate};
  if (backend->submitted) {
    return {false, "compute_pipeline_busy"};
  }
  if (backend->block_seed) {
    backend->seed_entered = true;
    backend->ready.notify_all();
    backend->ready.wait(lock,
                        [&] { return backend->stop || backend->seed_open; });
    if (backend->stop) {
      return {false, "compute_device_lost"};
    }
  }
  backend->seeded = generation;
  return {true, "ok"};
}

[[nodiscard]] rund::AccelCheck
FakeSubmit(const std::shared_ptr<void> &raw,
           const accel::BackendResidencySlidingDescriptor &descriptor,
           const accel::KernelCompletion completion, void *const user,
           const accel::KernelTiming, const accel::PipelineSubmitMode mode,
           const std::span<const std::uint32_t> locals) noexcept {
  const auto backend = std::static_pointer_cast<FakeSlidingBackend>(raw);
  if (backend == nullptr || completion == nullptr || user == nullptr ||
      mode != accel::PipelineSubmitMode::Residency || locals.empty()) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  bool inline_completion = false;
  bool reject_after_inline = false;
  bool cross_thread_early_completion = false;
  bool reject_after_cross_thread = false;
  bool block_submit_return = false;
  std::shared_ptr<FakeSlidingBackend> cross_backend{};
  void (*nested_submit)(void *) noexcept = nullptr;
  void *nested_user = nullptr;
  std::uint32_t generation = 0u;
  {
    std::lock_guard lock{backend->gate};
    const bool new_run = backend->descriptor_owner != descriptor.owner ||
                         backend->descriptor_plan != descriptor.plan_identity ||
                         backend->descriptor_token != descriptor.token ||
                         backend->descriptor_run != descriptor.generation;
    const bool coordinate =
        descriptor.owner != nullptr && descriptor.plan_identity != 0u &&
        descriptor.token != 0u && descriptor.generation != 0u &&
        descriptor.stride != 0u && descriptor.slot < descriptor.stride &&
        descriptor.coordinate ==
            descriptor.turn * descriptor.stride + descriptor.slot;
    const bool valid_generation =
        descriptor.descriptor_generation != 0u &&
        (new_run ||
         descriptor.descriptor_generation > backend->descriptor_generation);
    if (backend->submitted || locals.size() > backend->locals.size() ||
        !coordinate || !valid_generation ||
        (descriptor.read_mask == 0u && descriptor.write_mask == 0u)) {
      return {false, "compute_pipeline_busy"};
    }
    backend->descriptor_owner = descriptor.owner;
    backend->descriptor_plan = descriptor.plan_identity;
    backend->descriptor_token = descriptor.token;
    backend->descriptor_run = descriptor.generation;
    backend->descriptor_generation = descriptor.descriptor_generation;
    backend->completion = completion;
    backend->user = user;
    backend->local_count = locals.size();
    std::copy(locals.begin(), locals.end(), backend->locals.begin());
    ++backend->submission_count;
    backend->submitted = true;
    backend->released =
        backend->seeded != backend->delay_generation || backend->delay_open;
    inline_completion = backend->inline_completion;
    reject_after_inline = backend->reject_after_inline;
    cross_thread_early_completion = backend->cross_thread_early_completion;
    reject_after_cross_thread = backend->reject_after_cross_thread;
    block_submit_return = backend->block_submit_return;
    cross_backend = backend->cross_completion_backend;
    nested_submit = backend->nested_submit;
    nested_user = backend->nested_user;
    if (block_submit_return) {
      backend->submit_entered = true;
    }
    generation = backend->seeded + 1u;
  }
  if (nested_submit != nullptr) {
    nested_submit(nested_user);
  }
  if (inline_completion) {
    completion(user, FakeResult(generation, locals.size()));
    return reject_after_inline
               ? rund::AccelCheck{false, "compute_pipeline_busy"}
               : rund::AccelCheck{true, "ok"};
  }
  if (cross_thread_early_completion) {
    backend->worker = std::thread{
        [backend, completion, user, generation, local_count = locals.size()] {
          completion(user, FakeResult(generation, local_count));
          {
            std::lock_guard lock{backend->gate};
            backend->completion = nullptr;
            backend->user = nullptr;
            backend->submitted = false;
            backend->released = false;
            backend->early_completion_returned = true;
          }
          backend->ready.notify_all();
        }};
    bool completion_returned = false;
    {
      std::unique_lock lock{backend->gate};
      completion_returned =
          backend->ready.wait_for(lock, std::chrono::seconds{5}, [&] {
            return backend->early_completion_returned;
          });
    }
    if (!completion_returned) {
      return {false, "compute_device_lost"};
    }
    return reject_after_cross_thread
               ? rund::AccelCheck{false, "compute_pipeline_busy"}
               : rund::AccelCheck{true, "ok"};
  }
  if (cross_backend != nullptr) {
    accel::KernelCompletion cross_completion = nullptr;
    void *cross_user = nullptr;
    std::uint32_t cross_generation = 0u;
    std::size_t cross_local_count = 0u;
    bool cross_unknown = false;
    bool cross_known_failure = false;
    const char *cross_failure_reason = "compute_backend_failed";
    {
      std::lock_guard lock{cross_backend->gate};
      cross_completion = cross_backend->completion;
      cross_user = cross_backend->user;
      cross_generation = static_cast<std::uint32_t>(
          static_cast<std::int64_t>(cross_backend->seeded) + 1 +
          cross_backend->generation_delta);
      cross_local_count = cross_backend->local_count;
      cross_unknown = cross_backend->unknown_terminal;
      cross_known_failure = cross_backend->known_failure;
      cross_failure_reason = cross_backend->failure_reason;
      cross_backend->completion = nullptr;
      cross_backend->user = nullptr;
      cross_backend->submitted = false;
      cross_backend->released = false;
    }
    if (cross_completion != nullptr) {
      cross_completion(cross_user,
                       FakeResult(cross_generation, cross_local_count,
                                  cross_unknown, cross_known_failure,
                                  cross_failure_reason));
    }
  }
  backend->ready.notify_all();
  if (block_submit_return) {
    std::unique_lock lock{backend->gate};
    backend->ready.wait(lock,
                        [&] { return backend->stop || backend->submit_open; });
    if (backend->stop) {
      return {false, "compute_device_lost"};
    }
  }
  return {true, "ok"};
}

void BackendWorker(const std::shared_ptr<FakeSlidingBackend> &backend) {
  for (;;) {
    accel::KernelCompletion completion = nullptr;
    void *user = nullptr;
    std::uint32_t generation = 0u;
    std::size_t local_count = 0u;
    bool unknown = false;
    bool known_failure = false;
    const char *failure_reason = "compute_backend_failed";
    {
      std::unique_lock lock{backend->gate};
      backend->ready.wait(lock, [&] {
        return backend->stop || (backend->submitted && backend->released);
      });
      if (backend->stop) {
        return;
      }
      completion = backend->completion;
      user = backend->user;
      generation = static_cast<std::uint32_t>(
          static_cast<std::int64_t>(backend->seeded) + 1 +
          backend->generation_delta);
      local_count = backend->local_count;
      unknown = backend->unknown_terminal;
      known_failure = backend->known_failure;
      failure_reason = backend->failure_reason;
      backend->completion = nullptr;
      backend->user = nullptr;
      backend->submitted = false;
      backend->released = false;
    }
    completion(user, FakeResult(generation, local_count, unknown, known_failure,
                                failure_reason));
  }
}
} // namespace rund_node_test_pipeline_residency::native_sliding
