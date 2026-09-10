#pragma once

#include "src/accel/kernel/residency/device_vsm/terminal.hpp"
#include "src/compute/device/residency/execution/device_vsm/registration.hpp"
#include "src/compute/device/state.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>

namespace rund_node_test_pipeline_residency::device_vsm_test::wait_detail {

namespace accel = ::rund::node::accel::detail;
namespace residency = ::rund::compute::detail::residency;

// The native submitter retains only the request's raw callback user pointer.
// Keep a self-reference here so an accepted request owns every object needed
// by the callback until it returns.  A timed-out request intentionally keeps
// this cycle as the quarantine hold; it must never release its Authority lease
// while the backend may still call the raw pointer.
struct Owner final {
  explicit Owner(residency::DirectRecurrenceLease &&value) noexcept
      : lease(std::move(value)) {}

  accel::DeviceVsmRequest request{};
  accel::DeviceVsmSubmissionControl submission_control{};
  accel::DeviceVsmFinal final{};
  std::shared_ptr<rund::compute::detail::DeviceState> device{};
  std::shared_ptr<residency::Registry> registry{};
  std::shared_ptr<residency::DeviceVsmRegistration> registration{};
  residency::DirectRecurrenceLease lease{};
  std::mutex gate{};
  std::condition_variable done{};
  std::shared_ptr<Owner> self{};
  std::uint64_t callback_count{};
  bool abandoned{};
  bool final_ready{};
};

enum class Result : std::uint8_t { Failed, Closed, Quarantined };

struct Snapshot final {
  accel::DeviceVsmFinal final{};
  std::uint64_t callback_count{};
  bool abandoned{};
  bool final_ready{};
};

inline void complete(void *const raw, accel::DeviceVsmFinal &&final) noexcept {
  auto *const owner = static_cast<Owner *>(raw);
  if (owner == nullptr) {
    return;
  }
  std::shared_ptr<Owner> keep;
  bool notify = false;
  {
    std::lock_guard lock{owner->gate};
    keep = owner->self;
    if (owner->callback_count == 0u) {
      ++owner->callback_count;
      if (!owner->abandoned) {
        owner->final = std::move(final);
        owner->final_ready = true;
      }
      notify = true;
    }
  }
  if (notify) {
    owner->done.notify_all();
  }
}

[[nodiscard]] inline bool wait(Owner &owner) noexcept {
  std::unique_lock lock{owner.gate};
  return owner.done.wait_for(
      lock, std::chrono::seconds(5),
      [&owner] { return owner.callback_count == 1u; });
}

[[nodiscard]] inline std::uint64_t count(Owner &owner) noexcept {
  std::lock_guard lock{owner.gate};
  return owner.callback_count;
}

[[nodiscard]] inline Snapshot snapshot(Owner &owner) noexcept {
  std::lock_guard lock{owner.gate};
  return Snapshot{.final = owner.final,
                  .callback_count = owner.callback_count,
                  .abandoned = owner.abandoned,
                  .final_ready = owner.final_ready};
}

inline void abandon(Owner &owner) noexcept {
  std::lock_guard lock{owner.gate};
  owner.abandoned = true;
}

inline void release_self(Owner &owner) noexcept {
  std::lock_guard lock{owner.gate};
  owner.self.reset();
}

inline void quarantine(Owner &owner) noexcept {
  abandon(owner);
  if (owner.registry != nullptr && owner.lease) {
    static_cast<void>(owner.registry->authority().direct_recurrences().abort_direct_recurrence(
        owner.lease,
        ::rund::compute::Status::fail(::rund::compute::Reason::DeviceLost),
        residency::execution::TerminalKind::UnknownMayWrite, true));
  }
}

[[nodiscard]] inline Result dispose_unknown(Owner &owner,
                                            const std::uint64_t completed) noexcept {
  if (owner.registry == nullptr || owner.registration == nullptr ||
      !owner.lease) {
    return Result::Failed;
  }
  residency::DirectRecurrenceFinal prepared{};
  const bool prepared_ok =
      owner.registry->authority().direct_recurrences().prepare_direct_recurrence_final(
          owner.lease,
          ::rund::compute::Status::fail(::rund::compute::Reason::DeviceLost),
          residency::execution::TerminalKind::UnknownMayWrite, true, completed,
          prepared);
  if (!prepared_ok) {
    const auto aborted = owner.registry->authority().direct_recurrences().abort_direct_recurrence(
        owner.lease,
        ::rund::compute::Status::fail(::rund::compute::Reason::DeviceLost),
        residency::execution::TerminalKind::UnknownMayWrite, true);
    return aborted == residency::DirectAbort::Quarantined
               ? Result::Quarantined
               : Result::Failed;
  }
  const residency::ExecutionClose staged =
      owner.registry->authority().direct_recurrences().stage_direct_recurrence_final(
          std::move(prepared));
  const auto state = owner.registration->snapshot().state;
  const bool quarantined =
      staged.quarantined && state != nullptr &&
      state->phase() == residency::registration_detail::Lifecycle::Quarantined;
  const bool retained = owner.registration->release() ==
                        residency::RegistrationResult::Quarantined;
  return quarantined && retained ? Result::Quarantined : Result::Failed;
}

[[nodiscard]] inline bool cleanup_before_submit(Owner &owner) noexcept {
  bool clean = true;
  if (owner.registry != nullptr && owner.lease) {
    clean = owner.registry->authority().direct_recurrences().abort_direct_recurrence(
                owner.lease,
                ::rund::compute::Status::fail(
                    ::rund::compute::Reason::DeviceLost),
                residency::execution::TerminalKind::Known, false) ==
            residency::DirectAbort::Closed;
  }
  if (owner.registration != nullptr) {
    clean = owner.registration->release() == residency::RegistrationResult::Done &&
            clean;
  }
  if (clean) {
    release_self(owner);
  }
  return clean;
}

} // namespace rund_node_test_pipeline_residency::device_vsm_test::wait_detail
