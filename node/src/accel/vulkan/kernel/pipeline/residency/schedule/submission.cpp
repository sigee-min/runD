#include "internal.hpp"

#include "../../../../../clock.hpp"
#include "../../../../timeline/owner.hpp"

#include <limits>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace schedule {
namespace {

[[nodiscard]] bool
ValidRequest(const BackendResidencyScheduleRequest &request) noexcept {
  if (request.plan_identity == 0u || request.token == 0u ||
      request.generation == 0u || request.epoch_count == 0u ||
      request.role_count != ResidencyScheduleRoleCapacity ||
      request.release == nullptr || request.final == nullptr ||
      request.user == nullptr) {
    return false;
  }
  for (std::size_t role = 0u; role < request.role_count; ++role) {
    const BackendResidencyScheduleRole &value = request.roles[role];
    if (value.prepared == nullptr || value.role != role ||
        value.bank != role % 2u || value.first_control_generation == 0u ||
        value.control_generation_stride == 0u ||
        !UniqueLocals(value, value.local_count)) {
      return false;
    }
  }
  const std::size_t tail_role =
      static_cast<std::size_t>((request.epoch_count - 1u) % request.role_count);
  return UniqueLocals(request.roles[tail_role], request.tail_local_count);
}

} // namespace

bool Materialize(const BackendResidencyScheduleRequest &request,
                 std::array<VulkanResidencyScheduleRole,
                            ResidencyScheduleRoleCapacity> &roles,
                 VulkanResidencyScheduleRole &tail, VulkanAdapter *&adapter,
                 VulkanResidencyScheduleRun *&run) noexcept {
  adapter = nullptr;
  run = nullptr;
  for (std::size_t role = 0u; role < request.role_count; ++role) {
    const BackendResidencyScheduleRole &source = request.roles[role];
    auto *const pipeline = static_cast<VulkanPipeline *>(source.prepared.get());
    VulkanResidencySelection *const selection =
        pipeline == nullptr ? nullptr : pipeline->residency.get();
    if (!ValidVulkanPipeline(pipeline) || selection == nullptr ||
        !selection->ready || !selection->bounded ||
        selection->adapter != pipeline->adapter ||
        selection->active_window.load(std::memory_order_acquire) != nullptr ||
        selection->active_schedule.load(std::memory_order_acquire) != nullptr ||
        (adapter != nullptr && adapter != pipeline->adapter)) {
      return false;
    }
    adapter = pipeline->adapter;
    VulkanResidencyScheduleRole &target = roles[role];
    target.pipeline = pipeline;
    const rund::AccelCheck built = BuildVulkanResidencySubmission(
        *pipeline,
        std::span<const std::uint32_t>{source.locals.data(),
                                       source.local_count},
        target.commands, target.command_count, target.dispatch_count,
        target.control_count, target.reset_count, target.reset_bytes);
    if (!built.ok || target.command_count == 0u ||
        target.dispatch_count == 0u) {
      return false;
    }
    if (role == 0u) {
      run = &selection->schedule;
    }
  }
  const std::size_t tail_index =
      static_cast<std::size_t>((request.epoch_count - 1u) % request.role_count);
  const BackendResidencyScheduleRole &tail_source = request.roles[tail_index];
  tail.pipeline = roles[tail_index].pipeline;
  if (request.tail_local_count == tail_source.local_count) {
    tail = roles[tail_index];
  } else {
    const rund::AccelCheck built = BuildVulkanResidencySubmission(
        *tail.pipeline,
        std::span<const std::uint32_t>{tail_source.locals.data(),
                                       request.tail_local_count},
        tail.commands, tail.command_count, tail.dispatch_count,
        tail.control_count, tail.reset_count, tail.reset_bytes);
    if (!built.ok || tail.command_count == 0u || tail.dispatch_count == 0u) {
      return false;
    }
  }
  return adapter != nullptr && run != nullptr;
}

} // namespace schedule

using namespace schedule;

rund::AccelCheck SubmitVulkanResidencySchedule(
    const BackendResidencyScheduleRequest &request) noexcept {
  if (!ValidRequest(request)) {
    return Invalid();
  }
  std::array<VulkanResidencyScheduleRole, ResidencyScheduleRoleCapacity>
      roles{};
  VulkanResidencyScheduleRole tail{};
  VulkanAdapter *adapter = nullptr;
  VulkanResidencyScheduleRun *run = nullptr;
  if (!Materialize(request, roles, tail, adapter, run) ||
      adapter->residency_quarantined.load(std::memory_order_acquire)) {
    return Invalid();
  }
  std::vector<VulkanTimelineBatch> batches{};
  try {
    batches.resize(static_cast<std::size_t>(request.epoch_count));
  } catch (...) {
    return {false, "compute_pipeline_capacity"};
  }
  std::unique_lock run_lock{run->gate, std::try_to_lock};
  if (!run_lock.owns_lock() || run->active || run->quarantined) {
    return {false, "compute_pipeline_busy"};
  }
  std::unique_lock adapter_lock{adapter->mutex};
  std::unique_lock residency_lock{adapter->residency_mutex};
  if (adapter->residency_stop || adapter->residency_service != nullptr) {
    return Unavailable();
  }
  const auto reset_active = [&] {
    for (const BackendResidencyScheduleRole &source : request.roles) {
      auto *const pipeline =
          static_cast<VulkanPipeline *>(source.prepared.get());
      VulkanResidencyScheduleRun *expected = run;
      static_cast<void>(
          pipeline->residency->active_schedule.compare_exchange_strong(
              expected, nullptr, std::memory_order_acq_rel,
              std::memory_order_acquire));
    }
  };
  for (const BackendResidencyScheduleRole &source : request.roles) {
    auto *const pipeline = static_cast<VulkanPipeline *>(source.prepared.get());
    VulkanResidencyScheduleRun *expected = nullptr;
    if (!pipeline->residency->active_schedule.compare_exchange_strong(
            expected, run, std::memory_order_acq_rel,
            std::memory_order_acquire) &&
        expected != run) {
      reset_active();
      return {false, "compute_pipeline_busy"};
    }
  }
  std::uint32_t generation = 0u;
  rund::AccelCheck prepared =
      ReserveVulkanTimelineGeneration(adapter->timeline, generation);
  std::array<std::uint64_t, ResidencyScheduleRoleCapacity> ready_base{};
  std::uint64_t done_base = 0u;
  for (std::uint64_t epoch = 0u; prepared.ok && epoch < request.epoch_count;
       ++epoch) {
    VulkanTimelinePoint current{};
    prepared =
        ReserveVulkanTimelinePoint(adapter->timeline, generation, current);
    if (!prepared.ok) {
      break;
    }
    const std::size_t role =
        static_cast<std::size_t>(epoch % ResidencyScheduleRoleCapacity);
    if (epoch < ResidencyScheduleRoleCapacity) {
      ready_base[role] = current.ready_value;
    }
    if (epoch == 0u) {
      done_base = current.value;
    }
    VulkanResidencyScheduleRole &native =
        epoch + 1u == request.epoch_count ? tail : roles[role];
    batches[static_cast<std::size_t>(epoch)] = VulkanTimelineBatch{
        .commands = std::span<const VkCommandBuffer>{native.commands.data(),
                                                     native.command_count},
        .point = current,
    };
  }
  if (!prepared.ok) {
    if (generation != 0u) {
      static_cast<void>(
          CancelVulkanTimelineGeneration(adapter->timeline, generation));
    }
    ClearActive(*run);
    return prepared;
  }
  std::uint64_t queue_calls = 0u;
  const rund::AccelCheck submitted =
      SubmitVulkanTimelineStream(*adapter, batches, queue_calls, true);
  if (!submitted.ok || queue_calls != 1u) {
    static_cast<void>(
        CancelVulkanTimelineGeneration(adapter->timeline, generation));
    ClearActive(*run);
    return submitted.ok ? Unavailable() : submitted;
  }
  run->request = request;
  run->roles = roles;
  run->tail = tail;
  run->cells = {};
  run->ready_base = ready_base;
  run->done_base = done_base;
  run->timeline_generation = generation;
  run->release_count = 0u;
  run->signaled_count = 0u;
  run->inflight_peak = 0u;
  run->success_prefix = 0u;
  run->suppressed_first = 0u;
  run->suppressed_count = 0u;
  run->first_failure = {true, "ok"};
  run->submit_begin_ns = MonotonicNanoseconds();
  run->abort_failure = {};
  run->aborting = false;
  run->active = true;
  run->final_sent = false;
  adapter->residency_user = run;
  adapter->residency_service = Service;
  run_lock.unlock();
  residency_lock.unlock();
  adapter_lock.unlock();
  adapter->residency_cv.notify_one();
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
