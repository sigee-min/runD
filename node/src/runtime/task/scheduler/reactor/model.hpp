#pragma once

#include <rund/net/socket.hpp>

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
  std::uint64_t stop_source_id = 0u;
  std::uint64_t stop_generation = 0u;
  std::uint64_t stop_epoch = 0u;
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

struct ReactorRequest {
  std::uint64_t wait_id = 0u;
  std::uint64_t task_id = 0u;
  ReactorHandle fd = kInvalidReactorHandle;
  ReactorInterest interest = ReactorInterest::None;
};

struct ReactorReady {
  std::uint64_t wait_id = 0u;
  std::uint64_t task_id = 0u;
  ReactorHandle fd = kInvalidReactorHandle;
  ReactorInterest interest = ReactorInterest::None;
  ReactorEvent events = ReactorEvent::None;
  bool failed = false;
  bool invalid = false;
};

struct ReactorFdPreviousInterest {
  ReactorHandle fd = kInvalidReactorHandle;
  ReactorInterest interest = ReactorInterest::None;
};

struct ReactorProbeResult {
  bool failed = false;
  bool unavailable = false;
  ReactorReady ready{};
  bool has_ready = false;
};

struct ReactorFdState {
  ReactorHandle fd = kInvalidReactorHandle;
  std::uint32_t first_wait = kNoReactorSlot;
  std::uint32_t last_wait = kNoReactorSlot;
  std::uint32_t wait_count = 0u;
  std::uint32_t read_count = 0u;
  std::uint32_t write_count = 0u;
  ReactorInterest backend_interest = ReactorInterest::None;
  std::uint64_t fd_generation = 0u;
  std::uint64_t fd_device = 0u;
  std::uint64_t fd_inode = 0u;
  std::uint32_t fd_mode = 0u;
  ReactorHandle identity_guard = kInvalidReactorHandle;
  bool registered = false;
  bool remove_deferred = false;
  bool fd_identity_valid = false;
  bool batch_touched = false;
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
  std::uint32_t defer_depth = 0u;
  std::uint32_t batch_add_defer_depth = 0u;
  bool defer_registration_apply = false;
};

struct ReactorRuntime {
  ReactorPlatform platform{};
  ReactorApplyPolicy apply_policy{};
  ReactorRegistry registry{};
  std::vector<ReactorRegistrationChange> changes{};
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
