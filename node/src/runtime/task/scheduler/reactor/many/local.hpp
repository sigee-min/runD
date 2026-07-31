#pragma once

#include <rund/net/ready/many.hpp>

#include "../../../../../host/net/interest.hpp"
#include "../../../../../host/net/operation.hpp"
#include "../../../../reactor/platform.hpp"
#include "../../state/model/task.hpp"
#include "../../state/storage.hpp"
#include "../../timer/store.hpp"
#include "../backend.hpp"
#include "../backlog.hpp"
#include "../cleanup/request.hpp"
#include "../generation.hpp"
#include "../record.hpp"
#include "../registry.hpp"
#include "../state.hpp"
#include "../stats.hpp"
#include "../timeout.hpp"
#include "events.hpp"
#include "probe/raw.hpp"
#include "store.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <vector>

namespace rund::node {

struct ReadyManyEntry {
  TaskRecord *record = nullptr;
  std::uint64_t task_id = 0u;
  std::uint64_t group_id = 0u;
  std::uint32_t output_limit = 0u;
  std::span<const ReactorManyRequest> requests{};
  ReasonCode code = ReasonCode::TaskInvalid;

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == ReasonCode::Ok;
  }
};

[[nodiscard]] ::rund::net::ready::many::Wait
FailManyCode(ReasonCode code) noexcept;
[[nodiscard]] std::uint32_t
OutputLimit(std::span<::rund::net::ready::Event> out,
            ::rund::net::ready::many::Budget budget) noexcept;
[[nodiscard]] ReactorManyGroup *
FindManyGroupByTimerWaitId(std::vector<ReactorManyGroup> &groups,
                           std::uint64_t timer_wait_id) noexcept;
struct ReadyManyAccess {
  [[nodiscard]] static ReadyManyEntry PrepareEntry(
      Scheduler &scheduler, std::span<const ReactorManyRequest> requests,
      std::span<::rund::net::ready::Event> out,
      ::rund::net::ready::many::Budget budget, std::uint64_t stop_scheduler_id,
      std::uint64_t stop_source_id, std::uint64_t stop_generation,
      std::uint64_t stop_epoch) noexcept;
  [[nodiscard]] static ::rund::net::ready::many::Wait
  TryImmediate(Scheduler &scheduler, ReadyManyEntry &entry,
               std::span<::rund::net::ready::Event> out,
               std::optional<std::chrono::nanoseconds> timeout) noexcept;
  [[nodiscard]] static ::rund::net::ready::many::Wait
  Park(Scheduler &scheduler, ReadyManyEntry &entry,
       std::optional<std::chrono::nanoseconds> timeout,
       std::uint64_t stop_source_id, std::uint64_t stop_generation,
       std::uint64_t stop_epoch, std::uint64_t ready_set_id,
       std::uint64_t ready_set_generation) noexcept;
  [[nodiscard]] static bool
  ParkRegisterWaits(Scheduler &scheduler, ReadyManyEntry &entry,
                    std::uint64_t stop_source_id, std::uint64_t stop_generation,
                    std::uint64_t stop_epoch) noexcept;
  [[nodiscard]] static bool
  ParkRegisterTimeout(Scheduler &scheduler, ReadyManyEntry &entry,
                      std::uint64_t timer_wait_id, std::uint64_t stop_source_id,
                      std::uint64_t stop_generation, std::uint64_t stop_epoch,
                      const TimerDeadline &timer_deadline) noexcept;
  [[nodiscard]] static ::rund::net::ready::many::Wait
  Resume(Scheduler &scheduler, ReadyManyEntry &entry,
         std::span<::rund::net::ready::Event> out,
         std::uint64_t group_id) noexcept;
};

} // namespace rund::node
