#pragma once

#include <rund/net/socket.hpp>
#include <rund/task/cancel/identity.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "../../../reactor/platform.hpp"

namespace rund::node {

struct ReactorWait {
  ::rund::net::SocketView socket{};
  std::uint64_t task_id = 0u;
  std::uint64_t wait_id = 0u;
  std::uint64_t host_handle_id = 0u;
  std::uint64_t fd_generation = 0u;
  ::rund::detail::task::StopSourceIdentity stop{};
  ReactorHandle fd = kInvalidReactorHandle;
  ReactorInterest interest = ReactorInterest::None;
};

inline constexpr std::uint32_t kNoReactorSlot =
    std::numeric_limits<std::uint32_t>::max();

struct ReactorWaitSlot {
  ReactorWait wait{};
  std::uint32_t previous_fd = kNoReactorSlot;
  std::uint32_t next_fd = kNoReactorSlot;
};

enum class ReactorReadyDisposition : std::uint8_t {
  Ready,
  Invalid,
  PollFailed,
};

struct ReactorReady {
  std::uint64_t wait_id = 0u;
  std::uint64_t task_id = 0u;
  ReactorHandle fd = kInvalidReactorHandle;
  ReactorInterest interest = ReactorInterest::None;
  ReactorEvent events = ReactorEvent::None;
  ReactorReadyDisposition disposition = ReactorReadyDisposition::Ready;
};

struct ReactorFdPreviousInterest {
  ReactorHandle fd = kInvalidReactorHandle;
  ReactorInterest interest = ReactorInterest::None;
};

enum class ReactorFdRegistrationPhase : std::uint8_t {
  Idle,
  Active,
  DeferredRemove,
};

class ReactorFdRegistration final {
public:
  constexpr ReactorFdRegistration() noexcept = default;

  [[nodiscard]] static constexpr ReactorFdRegistration idle() noexcept {
    return ReactorFdRegistration{};
  }

  [[nodiscard]] static constexpr ReactorFdRegistration
  active(const ReactorInterest interest) noexcept {
    return interest == ReactorInterest::None
               ? idle()
               : ReactorFdRegistration{ReactorFdRegistrationPhase::Active,
                                       interest};
  }

  [[nodiscard]] static constexpr ReactorFdRegistration
  deferred_remove(const ReactorInterest interest) noexcept {
    return interest == ReactorInterest::None
               ? idle()
               : ReactorFdRegistration{
                     ReactorFdRegistrationPhase::DeferredRemove, interest};
  }

  [[nodiscard]] constexpr ReactorFdRegistrationPhase phase() const noexcept {
    return phase_;
  }

  [[nodiscard]] constexpr ReactorInterest interest() const noexcept {
    return interest_;
  }

  [[nodiscard]] constexpr bool is_idle() const noexcept {
    return phase_ == ReactorFdRegistrationPhase::Idle;
  }

private:
  constexpr ReactorFdRegistration(const ReactorFdRegistrationPhase phase,
                                  const ReactorInterest interest) noexcept
      : phase_(phase), interest_(interest) {}

  ReactorFdRegistrationPhase phase_ = ReactorFdRegistrationPhase::Idle;
  ReactorInterest interest_ = ReactorInterest::None;
};

struct ReactorFdState {
  ReactorHandle fd = kInvalidReactorHandle;
  std::uint32_t first_wait = kNoReactorSlot;
  std::uint32_t last_wait = kNoReactorSlot;
  std::uint32_t wait_count = 0u;
  std::uint32_t read_count = 0u;
  std::uint32_t write_count = 0u;
  ReactorFdRegistration registration = ReactorFdRegistration::idle();
  std::uint64_t fd_generation = 0u;
  ReactorPlatformHandleIdentity fd_identity =
      ReactorPlatformHandleIdentity::invalid();
  ReactorHandle identity_guard = kInvalidReactorHandle;
  bool batch_touched = false;

  [[nodiscard]] constexpr bool erasable() const noexcept {
    return wait_count == 0u && registration.is_idle() &&
           fd_identity.disposition() ==
               ReactorPlatformHandleIdentityDisposition::Invalid &&
           identity_guard == kInvalidReactorHandle;
  }
};

struct ReactorRegistry {
  std::vector<ReactorWaitSlot> slots{};
  std::vector<std::uint32_t> order{};
  std::vector<std::uint32_t> free_slots{};
  std::vector<ReactorFdState> fds{};
  std::size_t live = 0u;
  std::size_t deferred_removes = 0u;
};

struct ReactorApplyPolicy {
  std::uint32_t batch_scope_depth = 0u;
};

struct ReactorRuntime {
  ReactorPlatform platform{};
  ReactorApplyPolicy apply_policy{};
  ReactorRegistry registry{};
  std::vector<ReactorRegistrationChange> changes{};
  std::vector<ReactorPlatformReady> platform_ready{};
  std::vector<BatchIoReady> probe_ready{};
  std::vector<ReactorReady> ready{};
  std::vector<ReactorReady> ready_backlog{};
  std::vector<ReactorReady> ordered_ready_scratch{};
  std::vector<ReactorReady> budget_ready_scratch{};
  std::vector<ReactorReady> drain_ready_scratch{};
  std::vector<ReactorWait> removed_wait_scratch{};
  std::vector<ReactorWait> stale_wait_scratch{};
  std::vector<ReactorFdPreviousInterest> previous_interest_scratch{};
};

} // namespace rund::node
