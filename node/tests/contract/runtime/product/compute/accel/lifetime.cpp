#include "local.hpp"

#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"
#include "src/compute/virtual/state.hpp"

#include <rund/compute.hpp>
#include <rund/compute/session.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <thread>

namespace rund::node::test_contract {
namespace {

using Barrier =
    compute::detail::device_vsm_product_detail::DeviceVsmTestBarrier;
using Phase = compute::detail::device_vsm_product_detail::DeviceVsmTestPhase;
using State = compute::detail::VirtualPipelineState;

struct BarrierGuard final {
  std::shared_ptr<State> state{};
  Barrier *barrier{};
  compute::Submission *task{};
  bool done{};

  void arm(const std::shared_ptr<State> &value, Barrier &hook,
           compute::Submission &submission) noexcept {
    state = value;
    barrier = &hook;
    task = &submission;
    if (state != nullptr) {
      std::lock_guard lock{state->gate};
      state->device_vsm_test_barrier = barrier;
    }
  }

  [[nodiscard]] Phase wait_callback() noexcept {
    if (barrier == nullptr || task == nullptr) {
      return Phase::Failed;
    }
    for (;;) {
      const Phase phase = barrier->load();
      if (phase == Phase::Callback || phase == Phase::Failed) {
        return phase;
      }
      if (task->poll().completed) {
        barrier->publish(Phase::Failed);
        return Phase::Failed;
      }
      barrier->phase.wait(phase, std::memory_order_acquire);
    }
  }

  void release() noexcept {
    if (barrier == nullptr) {
      return;
    }
    if (wait_callback() == Phase::Callback) {
      barrier->publish(Phase::Released);
    }
  }

  void finish() noexcept { done = true; }

  ~BarrierGuard() noexcept {
    if (barrier != nullptr && !done) {
      Phase phase = barrier->load();
      if (phase == Phase::Accepted) {
        phase = wait_callback();
      }
      if (phase == Phase::Callback) {
        barrier->publish(Phase::Released);
        if (task != nullptr) {
          (void)task->wait();
        }
      } else if (phase == Phase::Released && task != nullptr) {
        (void)task->wait();
      } else if (phase == Phase::Armed) {
        barrier->publish(Phase::Failed);
      }
    }
    if (state != nullptr && barrier != nullptr) {
      std::lock_guard lock{state->gate};
      if (state->device_vsm_test_barrier == barrier) {
        state->device_vsm_test_barrier = nullptr;
      }
    }
  }
};

} // namespace

int CheckComputeAccelLifetime(::rund::Session &session,
                              const compute::Target target) {
  constexpr std::size_t ElementCount = 25u;
  constexpr std::size_t FrameElements = 16u;
  constexpr std::array<std::uint64_t, ElementCount> first = [] {
    std::array<std::uint64_t, ElementCount> values{};
    for (std::size_t index = 0u; index < values.size(); ++index) {
      values[index] = index + 1u;
    }
    return values;
  }();
  constexpr std::array<std::uint64_t, ElementCount> second = [] {
    std::array<std::uint64_t, ElementCount> values{};
    for (std::size_t index = 0u; index < values.size(); ++index) {
      values[index] = 100u + index;
    }
    return values;
  }();
  constexpr std::array<std::uint64_t, ElementCount> expected = [] {
    std::array<std::uint64_t, ElementCount> values{};
    for (std::size_t index = 0u; index < values.size(); ++index) {
      values[index] = index + 1u + 100u + index;
    }
    return values;
  }();
  compute::Submission retained{};
  std::shared_ptr<compute::VirtualBacking> output_store{};
  std::shared_ptr<State> state{};
  std::uint64_t version_before{};
  Barrier barrier{};
  BarrierGuard guard{};
  {
    auto device = compute::open(target);
    if (!device) {
      return 1;
    }
    auto program = compute::on(*device)
                       .input<std::uint64_t>(FrameElements)
                       .zip_input<std::uint64_t>(FrameElements)
                       .map("node-device-vsm-async-lifetime",
                            [](auto left, auto right) { return left + right; })
                       .compile();
    if (!program) {
      return 2;
    }
    auto first_backing =
        compute::resident_virtual_backing<std::uint64_t>(*device, ElementCount);
    auto second_backing =
        compute::resident_virtual_backing<std::uint64_t>(*device, ElementCount);
    auto output_backing =
        compute::resident_virtual_backing<std::uint64_t>(*device, ElementCount);
    if (!first_backing || !second_backing || !output_backing ||
        !(*first_backing)->write(0u, std::as_bytes(std::span{first})) ||
        !(*second_backing)->write(0u, std::as_bytes(std::span{second}))) {
      return 3;
    }
    output_store = *output_backing;
    version_before =
        compute::detail::VirtualBackingAccess::version(*output_store);
    auto first_buffer =
        compute::virtual_buffer<std::uint64_t>(ElementCount, *first_backing);
    auto second_buffer =
        compute::virtual_buffer<std::uint64_t>(ElementCount, *second_backing);
    auto output_buffer =
        compute::virtual_buffer<std::uint64_t>(ElementCount, *output_backing);
    if (!first_buffer || !second_buffer || !output_buffer) {
      return 4;
    }
    auto pipeline = compute::virtual_pipeline(*program, *first_buffer,
                                              *second_buffer, *output_buffer);
    if (!pipeline) {
      return 5;
    }
    state = compute::detail::VirtualPipelineAccess::state(*pipeline);
    if (state == nullptr || !state->geometry.device_vsm_required) {
      return 6;
    }
    guard.arm(state, barrier, retained);
    retained = session.compute(*pipeline).submit();
  }
  if (guard.wait_callback() == Phase::Failed) {
    return 7;
  }
  for (;;) {
    const compute::Poll poll = retained.poll();
    if (poll.submitted && poll.backend_submitted && !poll.completed) {
      break;
    }
    if (poll.completed) {
      return 8;
    }
    std::this_thread::yield();
  }
  guard.release();
  const compute::Completion result = retained.wait();
  const compute::Poll poll = retained.poll();
  const compute::Stats stats = result.stats();
  const auto owner =
      state == nullptr
          ? std::shared_ptr<compute::detail::device_vsm_product_detail::
                                DeviceVsmProductOwner>{}
          : std::static_pointer_cast<
                compute::detail::device_vsm_product_detail::
                    DeviceVsmProductOwner>(state->device_vsm_product_cache);
  std::array<std::uint64_t, ElementCount> observed{};
  const bool private_ok =
      owner != nullptr && owner->submitted && owner->evidence != nullptr &&
      owner->evidence->final_received &&
      owner->evidence->backing_publication_count == 1u &&
      owner->evidence->authority_accept_count == 1u &&
      owner->evidence->pipeline_terminal_count == owner->pipeline_count &&
      owner->evidence->native.native_submit_count == 1u &&
      owner->evidence->native.host_epoch_callback_count == 0u &&
      owner->evidence->native.host_service_turn_count == 0u;
  const bool public_ok =
      stats.backend == target.backend() && poll.completed &&
      poll.backend_submitted && stats.command_submits == 1u &&
      stats.dispatches == 1u && stats.final_dispatches == 1u &&
      stats.publication.commit_count == 0u &&
      stats.publication.discard_count == 0u;
  const bool output_ok =
      output_store != nullptr &&
      output_store->read(0u, std::as_writable_bytes(std::span{observed})) &&
      observed == expected &&
      compute::detail::VirtualBackingAccess::version(*output_store) ==
          version_before + 1u &&
      compute::detail::VirtualBackingAccess::recovery_bytes(*output_store) ==
          0u;
  if (!(result && private_ok && public_ok && output_ok)) {
    std::fprintf(
        stderr,
        "async lifetime backend=%u result=%u private=%u public=%u "
        "output=%u owner=%u submitted=%u final=%u native=%llu "
        "host_cb=%llu host_turn=%llu submits=%llu dispatches=%llu "
        "final_dispatches=%llu commit=%llu discard=%llu\n",
        static_cast<unsigned>(target.backend()),
        static_cast<unsigned>(!!result), static_cast<unsigned>(private_ok),
        static_cast<unsigned>(public_ok), static_cast<unsigned>(output_ok),
        static_cast<unsigned>(owner != nullptr),
        static_cast<unsigned>(owner != nullptr && owner->submitted),
        static_cast<unsigned>(owner != nullptr && owner->evidence != nullptr &&
                              owner->evidence->final_received),
        static_cast<unsigned long long>(
            owner != nullptr && owner->evidence != nullptr
                ? owner->evidence->native.native_submit_count
                : 0u),
        static_cast<unsigned long long>(
            owner != nullptr && owner->evidence != nullptr
                ? owner->evidence->native.host_epoch_callback_count
                : 0u),
        static_cast<unsigned long long>(
            owner != nullptr && owner->evidence != nullptr
                ? owner->evidence->native.host_service_turn_count
                : 0u),
        static_cast<unsigned long long>(stats.command_submits),
        static_cast<unsigned long long>(stats.dispatches),
        static_cast<unsigned long long>(stats.final_dispatches),
        static_cast<unsigned long long>(stats.publication.commit_count),
        static_cast<unsigned long long>(stats.publication.discard_count));
    std::fprintf(
        stderr,
        "async private backing_pub=%llu authority=%llu terminal=%llu "
        "version=%llu/%llu recovery=%llu\n",
        static_cast<unsigned long long>(
            owner != nullptr && owner->evidence != nullptr
                ? owner->evidence->backing_publication_count
                : 0u),
        static_cast<unsigned long long>(
            owner != nullptr && owner->evidence != nullptr
                ? owner->evidence->authority_accept_count
                : 0u),
        static_cast<unsigned long long>(
            owner != nullptr && owner->evidence != nullptr
                ? owner->evidence->pipeline_terminal_count
                : 0u),
        static_cast<unsigned long long>(version_before),
        static_cast<unsigned long long>(
            output_store == nullptr
                ? 0u
                : compute::detail::VirtualBackingAccess::version(
                      *output_store)),
        static_cast<unsigned long long>(
            output_store == nullptr
                ? 0u
                : compute::detail::VirtualBackingAccess::recovery_bytes(
                      *output_store)));
    std::fprintf(
        stderr,
        "async completion reason=%u code=%u error=%.*s "
        "mapped=%u native_ok=%u native_reason=%llu native_code=%u "
        "result_acquired=%u terminal=%u final_reason=%u final_code=%u "
        "may_write=%u quarantined=%u\n",
        static_cast<unsigned>(result.reason()),
        static_cast<unsigned>(result.code()),
        static_cast<int>(result.error().size()), result.error().data(),
        static_cast<unsigned>(owner != nullptr && owner->evidence != nullptr &&
                              owner->evidence->native.result_mapped),
        static_cast<unsigned>(owner != nullptr && owner->evidence != nullptr &&
                              owner->evidence->native.native_check_ok),
        static_cast<unsigned long long>(
            owner != nullptr && owner->evidence != nullptr
                ? owner->evidence->native.native_check_reason
                : 0u),
        static_cast<unsigned>(owner != nullptr && owner->evidence != nullptr
                                  ? owner->evidence->native.native_check_code
                                  : 0u),
        static_cast<unsigned>(owner != nullptr && owner->evidence != nullptr &&
                              owner->evidence->native.result_acquired),
        static_cast<unsigned>(
            owner != nullptr && owner->evidence != nullptr
                ? owner->evidence->final_terminal
                : node::accel::detail::DeviceVsmTerminal::Known),
        static_cast<unsigned>(owner != nullptr && owner->evidence != nullptr
                                  ? owner->evidence->final_reason
                                  : compute::Reason::CompletionInvalid),
        static_cast<unsigned>(owner != nullptr && owner->evidence != nullptr
                                  ? owner->evidence->final_code
                                  : compute::Code::Execution),
        static_cast<unsigned>(owner != nullptr && owner->evidence != nullptr &&
                              owner->evidence->native.may_write),
        static_cast<unsigned>(owner != nullptr && owner->evidence != nullptr &&
                              owner->evidence->quarantined));
  }
  guard.finish();
  return result && private_ok && public_ok && output_ok ? 0 : 9;
}

} // namespace rund::node::test_contract
