#pragma once

#include <rund/task/cancel.hpp>
#include <rund/task/handle.hpp>
#include <rund/session/scheduler.hpp>

namespace rund::node {

class Scheduler;

namespace scheduler_access {

[[nodiscard]] Scheduler *ActiveScheduler() noexcept;
[[nodiscard]] ::rund::SchedulerConfig ActiveLimits() noexcept;
[[nodiscard]] std::uint32_t
CoroutineFrameByteLimit(const Scheduler &scheduler) noexcept;
[[nodiscard]] ::rund::detail::task::StopIdentity
StopTokenIdentity(task::stop_token token) noexcept;

} // namespace scheduler_access

} // namespace rund::node
