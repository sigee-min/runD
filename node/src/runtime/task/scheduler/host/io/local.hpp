#pragma once

#include "../../state/storage/host/slot.hpp"
#include "../../state/storage.hpp"

#include "../../../../../host/io/access.hpp"
#include "../../../../../host/io/validation.hpp"
#include "../../../../platform/io.hpp"

namespace rund::node::host_io {

[[nodiscard]] ::rund::host::io::ReadResult
ReadResult(HostIoCompletion completion) noexcept;

[[nodiscard]] ::rund::host::io::WriteResult
WriteResult(HostIoCompletion completion) noexcept;

[[nodiscard]] ReasonCode OutcomeCode(const HostIoOutcome &outcome) noexcept;

[[nodiscard]] ::rund::host::Status EventStatus(ReasonCode code) noexcept;

[[nodiscard]] HostIoSlot *Claim(Scheduler &scheduler, SchedulerState &state,
                                const HostIoOperation &operation) noexcept;

void Queue(SchedulerHostIoState &state, HostIoSlot &slot) noexcept;
void Release(SchedulerHostIoState &state, HostIoSlot &slot) noexcept;

[[nodiscard]] bool Valid(SchedulerHostIoState &state, const HostIoSlot *slot,
                         HostIoKind kind) noexcept;

void Run(Scheduler &scheduler, SchedulerState &state);

} // namespace rund::node::host_io
