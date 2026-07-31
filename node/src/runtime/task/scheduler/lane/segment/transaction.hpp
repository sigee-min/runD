#pragma once

#include "../../state/storage.hpp"

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace rund::node::segment {

[[nodiscard]] inline bool Idle(const TaskLane &lane) noexcept {
  return !lane.stop && !lane.has_job && !lane.running &&
         !lane.nested_job_active &&
         !lane.root_reserved.load(std::memory_order_acquire) &&
         lane.completed_job_sequence.load(std::memory_order_acquire) == 0u &&
         lane.work_head == nullptr && lane.external_wake_head == nullptr &&
         lane.direct_ready_head == nullptr &&
         lane.mailbox_state.load(std::memory_order_acquire) == 0u;
}

class Locks final {
public:
  explicit Locks(
      const std::vector<std::unique_ptr<TaskLane>> &lanes) noexcept
      : lanes_(&lanes) {
    for (std::size_t index = 0u; index < lanes.size(); ++index) {
      lanes[index]->mutex.lock();
      locked_through_ = index + 1u;
    }
  }

  Locks(const Locks &) = delete;
  Locks &operator=(const Locks &) = delete;

  ~Locks() {
    for (std::size_t reverse = locked_through_; reverse > 0u; --reverse) {
      (*lanes_)[reverse - 1u]->mutex.unlock();
    }
  }

private:
  const std::vector<std::unique_ptr<TaskLane>> *lanes_ = nullptr;
  std::size_t locked_through_ = 0u;
};

template <typename Active, typename Publisher>
[[nodiscard]] bool
Publish(const std::vector<std::unique_ptr<TaskLane>> &lanes,
        Active &&active,
        Publisher &&publisher) noexcept {
  static_assert(noexcept(std::declval<Active &>()(std::size_t{})),
                "lane admission projection must not throw");
  static_assert(noexcept(std::declval<Publisher &>()()),
                "lane segment publication must not throw");
  for (std::size_t index = 0u; index < lanes.size(); ++index) {
    if (lanes[index] == nullptr) {
      return false;
    }
  }

  Locks locks{lanes};
  for (std::size_t index = 0u; index < lanes.size(); ++index) {
    if (active(index) && !Idle(*lanes[index])) {
      return false;
    }
  }
  return std::forward<Publisher>(publisher)();
}

} // namespace rund::node::segment
