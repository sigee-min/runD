#pragma once

#include <cstdint>

#include "../../state/model/task.hpp"
#include "../../state/storage.hpp"
#include "request.hpp"

namespace rund::node::reactor_cancel_cleanup {

void WakeTask(Scheduler& scheduler, TaskRecord& record, std::uint64_t wait_id,
               ReactorHandle handle, ReactorInterest interest,
               ReactorEvent events,
               std::int64_t deadline_ns,
               ReasonCode reason, bool cleanup_ok) noexcept;

[[nodiscard]] bool
CancelTimeoutTimer(Scheduler &scheduler, std::uint64_t wait_id,
                   ReactorTimeoutCleanupPolicy policy) noexcept;

[[nodiscard]] bool CleanupGroup(Scheduler& scheduler,
                                const ReactorCleanupRequest& request) noexcept;

[[nodiscard]] bool CleanupSingleWait(Scheduler& scheduler,
                                     ReactorCleanupRequest request) noexcept;

}  // namespace rund::node::reactor_cancel_cleanup
