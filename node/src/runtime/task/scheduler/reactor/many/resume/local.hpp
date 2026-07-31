#pragma once

#include "../local.hpp"

namespace rund::node {

[[nodiscard]] bool CleanupReadyManyResume(Scheduler &scheduler,
                                          std::uint64_t group_id,
                                          ReasonCode ready_code) noexcept;
[[nodiscard]] ReasonCode ReadyManyResumeResultCode(ReasonCode ready_code,
                                                   bool cleanup_ok) noexcept;
[[nodiscard]] ::rund::net::ready::many::Result
MakeReadyManyCancelledResult() noexcept;
[[nodiscard]] ::rund::net::ready::many::Result
MakeReadyManyResumeResult(ReasonCode result_code, std::uint32_t copied,
                          bool budget_exhausted) noexcept;
void ResetReadyManyResumeTask(TaskRecord &record) noexcept;
void ResetReadyManyResumeWaitState(TaskRecord &record) noexcept;

} // namespace rund::node
