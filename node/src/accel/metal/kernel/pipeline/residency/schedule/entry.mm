#include "internal.hpp"

#include "../../../../../clock.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <new>
#include <utility>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK) &&                 \
    defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

namespace schedule_detail {

bool add(const std::uint64_t left, const std::uint64_t right,
         std::uint64_t &result) noexcept {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    return false;
  }
  result = left + right;
  return true;
}

bool same_role(const BackendResidencyScheduleRole &left,
               const BackendResidencyScheduleRole &right) {
  return left.prepared == right.prepared && left.locals == right.locals &&
         left.local_count == right.local_count &&
         left.first_control_generation == right.first_control_generation &&
         left.control_generation_stride == right.control_generation_stride &&
         left.role == right.role && left.bank == right.bank;
}

bool control_generation(const BackendResidencyScheduleRole &role,
                        const std::uint64_t epoch,
                        std::uint32_t &generation) noexcept {
  const std::uint64_t turn = epoch / ResidencyScheduleRoleCapacity;
  if (turn != 0u && role.control_generation_stride >
                        std::numeric_limits<std::uint64_t>::max() / turn) {
    return false;
  }
  const std::uint64_t delta = turn * role.control_generation_stride;
  if (delta > std::numeric_limits<std::uint32_t>::max() ||
      role.first_control_generation >
          std::numeric_limits<std::uint32_t>::max() - delta) {
    return false;
  }
  generation =
      role.first_control_generation + static_cast<std::uint32_t>(delta);
  return true;
}

bool valid_roles(const std::span<const BackendResidencyScheduleRole> roles,
                 const std::uint64_t epoch_count,
                 MetalAdapter *&adapter) noexcept {
  adapter = nullptr;
  if (roles.size() != ResidencyScheduleRoleCapacity || epoch_count <= 4u) {
    return false;
  }
  for (std::size_t index = 0u; index < roles.size(); ++index) {
    const BackendResidencyScheduleRole &role = roles[index];
    auto *const sequence = static_cast<MetalSequence *>(role.prepared.get());
    if (role.role != index || role.bank != index % 2u ||
        role.first_control_generation == 0u ||
        role.control_generation_stride == 0u || role.local_count == 0u ||
        role.local_count > ResidencyWindowLocalCapacity ||
        !ValidMetalSequence(sequence) || sequence->adapter == nullptr ||
        sequence->direct_aggregate || sequence->state_count != 0u ||
        sequence->guard_zero == nil ||
        [sequence->guard_zero contents] == nullptr ||
        !sequence->residency_selectable ||
        !sequence->residency_submission.ready() ||
        !sequence->residency_window.ready_for_submit() ||
        sequence->residency_window.service == nil ||
        MetalResidencyScheduleQueue(*sequence) == nil ||
        (adapter != nullptr && adapter != sequence->adapter)) {
      return false;
    }
    adapter = sequence->adapter;
  }
  return adapter != nullptr;
}

std::shared_ptr<MetalResidencyScheduleOwner>
owner_of(const std::shared_ptr<void> &opaque) noexcept {
  auto owner = std::static_pointer_cast<MetalResidencyScheduleOwner>(opaque);
  return owner != nullptr && owner->magic == ScheduleMagic ? std::move(owner)
                                                           : nullptr;
}

} // namespace schedule_detail

#endif

#endif

BackendResidencySchedulePreparation PrepareMetalResidencySchedule(
    const std::span<const BackendResidencyScheduleRole> roles,
    const std::uint64_t epoch_count,
    const std::size_t tail_local_count) noexcept {
  const rund::AccelCheck unavailable{false, "accel_metal_command_unavailable"};
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK) &&                 \
    defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  if (@available(macOS 26.0, iOS 26.0, *)) {
    MetalAdapter *adapter = nullptr;
    if (!schedule_detail::valid_roles(roles, epoch_count, adapter) ||
        tail_local_count == 0u ||
        tail_local_count >
            roles[(epoch_count - 1u) % roles.size()].local_count ||
        epoch_count > std::numeric_limits<std::size_t>::max()) {
      return {};
    }
    std::shared_ptr<schedule_detail::MetalResidencyScheduleOwner> owner;
    try {
      owner = std::make_shared<schedule_detail::MetalResidencyScheduleOwner>();
      owner->commands.resize(static_cast<std::size_t>(epoch_count));
    } catch (const std::bad_alloc &) {
      return BackendResidencySchedulePreparation{.check = unavailable};
    }
    owner->adapter = adapter;
    std::copy(roles.begin(), roles.end(), owner->roles.begin());
    owner->tail_local_count = tail_local_count;
    auto *const leader = static_cast<MetalSequence *>(roles[0u].prepared.get());
    owner->queue = MetalResidencyScheduleQueue(*leader);
    owner->service = dispatch_queue_create("rund.metal.residency.schedule",
                                           DISPATCH_QUEUE_SERIAL);
    id<MTLDevice> const device = (__bridge id<MTLDevice>)adapter->device.get();
    std::uint64_t retained = sizeof(schedule_detail::MetalResidencyScheduleOwner);
    const std::uint64_t host_commands =
        static_cast<std::uint64_t>(owner->commands.capacity()) *
        sizeof(schedule_detail::MetalResidencyScheduleCommand);
    if (!schedule_detail::add(retained, host_commands, retained)) {
      return BackendResidencySchedulePreparation{
          .check = {false, "compute_pipeline_capacity"}};
    }
    for (std::uint64_t epoch = 0u; epoch < epoch_count; ++epoch) {
      const BackendResidencyScheduleRole &role =
          roles[static_cast<std::size_t>(epoch % roles.size())];
      schedule_detail::MetalResidencyScheduleCommand &entry =
          owner->commands[epoch];
      entry.sequence = static_cast<MetalSequence *>(role.prepared.get());
      entry.epoch = epoch;
      entry.allocator = [device newCommandAllocator];
      entry.command = [device newCommandBuffer];
      if (owner->service == nil || entry.allocator == nil ||
          entry.command == nil ||
          !schedule_detail::control_generation(role, epoch,
                                               entry.control_generation)) {
        return BackendResidencySchedulePreparation{.check = unavailable};
      }
      const std::size_t local_count =
          epoch + 1u == epoch_count ? tail_local_count : role.local_count;
      const rund::AccelCheck encoded = EncodeMetalResidencyScheduleCommand(
          *entry.sequence, entry.allocator, entry.command,
          std::span<const std::uint32_t>{role.locals.data(), local_count},
          entry.dispatch_count, entry.control_count, entry.reset_count,
          entry.reset_bytes);
      if (!encoded.ok ||
          !schedule_detail::add(
              retained, static_cast<std::uint64_t>(entry.allocator.allocatedSize),
              retained)) {
        return BackendResidencySchedulePreparation{.check = encoded};
      }
    }
    owner->retained_bytes = retained;
    return BackendResidencySchedulePreparation{
        .check = {true, "ok"},
        .kind = BackendResidencyScheduleLowering::MetalPending,
        .retained_bytes = retained,
        .transient_bytes = 0u,
        .queue_calls = epoch_count,
        .owner = owner,
        .callbacks_async = true,
    };
  }
#else
  static_cast<void>(roles);
  static_cast<void>(epoch_count);
  static_cast<void>(tail_local_count);
#endif
  return BackendResidencySchedulePreparation{.check = unavailable};
}

rund::AccelCheck SubmitMetalResidencySchedule(
    const BackendResidencyScheduleRequest &request) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK) &&                 \
    defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  if (@available(macOS 26.0, iOS 26.0, *)) {
    const std::shared_ptr<schedule_detail::MetalResidencyScheduleOwner> owner =
        schedule_detail::owner_of(request.lowering);
    MetalAdapter *adapter = nullptr;
    if (owner == nullptr || request.plan_identity == 0u ||
        request.token == 0u || request.generation == 0u ||
        request.epoch_count != owner->commands.size() ||
        request.role_count != ResidencyScheduleRoleCapacity ||
        request.release == nullptr || request.final == nullptr ||
        request.user == nullptr ||
        !schedule_detail::valid_roles(request.roles, request.epoch_count,
                                       adapter) ||
        adapter != owner->adapter) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    for (std::size_t role = 0u; role < request.role_count; ++role) {
      if (!schedule_detail::same_role(owner->roles[role], request.roles[role])) {
        return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
      }
    }
    if (request.tail_local_count != owner->tail_local_count) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    std::unique_lock terminal_lock{adapter->residency_terminal_gate};
    std::unique_lock lock{owner->gate};
    if (owner->active || owner->final_sent ||
        adapter->residency_quarantined.load(std::memory_order_acquire)) {
      return rund::AccelCheck{false, "compute_pipeline_busy"};
    }
    for (std::size_t role = 0u; role < request.role_count; ++role) {
      auto *const sequence =
          static_cast<MetalSequence *>(request.roles[role].prepared.get());
      if (!sequence->residency_schedule.expired()) {
        return rund::AccelCheck{false, "compute_pipeline_busy"};
      }
    }
    std::array<MetalSequence *, ResidencyScheduleRoleCapacity> sequences{};
    std::array<std::uint64_t, ResidencyScheduleRoleCapacity> occurrences{};
    std::size_t sequence_count = 0u;
    for (const schedule_detail::MetalResidencyScheduleCommand &entry :
         owner->commands) {
      std::size_t index = 0u;
      while (index < sequence_count && sequences[index] != entry.sequence) {
        ++index;
      }
      if (index == sequence_count) {
        if (sequence_count == sequences.size()) {
          return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
        }
        sequences[sequence_count++] = entry.sequence;
      }
      ++occurrences[index];
    }
    for (std::size_t index = 0u; index < sequence_count; ++index) {
      if (sequences[index]->residency_window.event_value >
          TerminalCell::MaxGeneration - occurrences[index]) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
    }
    owner->request = request;
    owner->request.lowering.reset();
    owner->release_count = 0u;
    owner->success_prefix = 0u;
    owner->suppressed_first = 0u;
    owner->suppressed_count = 0u;
    owner->native_inflight = 0u;
    owner->native_inflight_peak = 0u;
    owner->submit_begin_ns = MonotonicNanoseconds();
    owner->first_failure = {true, "ok"};
    owner->active = true;
    owner->aborting = false;
    owner->unknown = false;
    owner->final_sent = false;
    const bool force_device_lost =
        adapter->device_loss_fault.take(SubmitKind::Work);
    const bool force_terminal_loss =
        adapter->fault_residency_terminal_once.exchange(
            false, std::memory_order_acq_rel);
    for (std::size_t role = 0u; role < request.role_count; ++role) {
      auto *const sequence =
          static_cast<MetalSequence *>(request.roles[role].prepared.get());
      sequence->residency_schedule = owner;
    }
    for (std::uint64_t epoch = 0u; epoch < request.epoch_count; ++epoch) {
      schedule_detail::MetalResidencyScheduleCommand &entry =
          owner->commands[epoch];
      const std::uint64_t value =
          ++entry.sequence->residency_window.event_value;
      entry.ready_value = value;
      entry.done_value = value;
      entry.force_device_lost =
          force_device_lost && epoch + 1u == request.epoch_count;
      entry.force_terminal_loss =
          force_terminal_loss && epoch + 1u == request.epoch_count;
      const std::weak_ptr<schedule_detail::MetalResidencyScheduleOwner> weak{
          owner};
      if (!entry.force_terminal_loss) {
        [entry.sequence->residency_window.done
            notifyListener:entry.sequence->residency_window.listener
                   atValue:value
                     block:^(id<MTLSharedEvent>, std::uint64_t observed) {
                       const std::shared_ptr<
                           schedule_detail::MetalResidencyScheduleOwner>
                           retained = weak.lock();
                       if (retained != nullptr && observed >= value) {
                         schedule_detail::complete(
                             retained, epoch, NativeTerminal::Known);
                       }
                     }];
      }
      [owner->queue waitForEvent:entry.sequence->residency_window.ready
                           value:value];
      const id<MTL4CommandBuffer> commands[] = {entry.command};
      [owner->queue commit:commands count:1u];
      if (!entry.force_terminal_loss) {
        [owner->queue signalEvent:entry.sequence->residency_window.done
                            value:value];
      }
    }
    return rund::AccelCheck{true, "ok"};
  }
#else
  static_cast<void>(request);
#endif
  return rund::AccelCheck{false, "accel_metal_command_unavailable"};
}

} // namespace rund::node::accel::detail
