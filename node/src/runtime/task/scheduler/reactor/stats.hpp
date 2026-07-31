#pragma once

#include <rund/task/stats/storage.hpp>

#include <cstddef>
#include <cstdint>

namespace rund::node {

void RecordReactorWaitRegistered(
    ::rund::detail::task::StatStorage &stats) noexcept;
void RecordReactorWaitsCanceled(::rund::detail::task::StatStorage &stats,
                                std::uint64_t count) noexcept;
void RecordReactorTimedWaitRegistered(
    ::rund::detail::task::StatStorage &stats) noexcept;
void RecordReactorTimeoutTimerCancel(
    ::rund::detail::task::StatStorage &stats) noexcept;
void RecordReactorTimeoutCleanupFailure(
    ::rund::detail::task::StatStorage &stats) noexcept;
void RecordReactorReadyEvents(::rund::detail::task::StatStorage &stats,
                              std::size_t count) noexcept;
void RecordReactorReadyManyRequest(
    ::rund::detail::task::StatStorage &stats) noexcept;
void RecordReactorReadyManyEvents(::rund::detail::task::StatStorage &stats,
                                  std::size_t count) noexcept;
void RecordReactorReadySetCreate(
    ::rund::detail::task::StatStorage &stats) noexcept;
void RecordReactorReadySetDestroy(
    ::rund::detail::task::StatStorage &stats) noexcept;
void RecordReactorReadySetClear(
    ::rund::detail::task::StatStorage &stats) noexcept;
void RecordReactorReadySetMemberAdded(
    ::rund::detail::task::StatStorage &stats) noexcept;
void RecordReactorReadySetMemberRemoved(
    ::rund::detail::task::StatStorage &stats) noexcept;
void RecordReactorReadySetWait(
    ::rund::detail::task::StatStorage &stats) noexcept;
void RecordReactorReadySetEvents(::rund::detail::task::StatStorage &stats,
                                 std::size_t count) noexcept;
void RecordReactorReadySetInvalidations(
    ::rund::detail::task::StatStorage &stats, std::size_t count) noexcept;
void RecordReactorCloseInvalidatedWaits(
    ::rund::detail::task::StatStorage &stats, std::size_t count) noexcept;
void RecordReactorBacklogPush(::rund::detail::task::StatStorage &stats,
                              std::size_t count) noexcept;
void RecordReactorBacklogDrain(::rund::detail::task::StatStorage &stats,
                               std::size_t count) noexcept;
void RecordReactorBacklogDepth(::rund::detail::task::StatStorage &stats,
                               std::size_t depth) noexcept;
void RecordReactorRegistrationApplyCall(
    ::rund::detail::task::StatStorage &stats) noexcept;
void RecordReactorRegistrationChangeApplied(
    ::rund::detail::task::StatStorage &stats) noexcept;

} // namespace rund::node
