#pragma once

#include <rund/task/operation/kind.hpp>

#include <rund/task/active.hpp>
#include <rund/task/handle.hpp>
#include <rund/task/status.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rund::detail::task {

struct ChannelDecision final {
  ::rund::task::Status status{
      ::rund::task::Status::fail(ReasonCode::ChannelInvalid)};
  bool complete_committed = false;
  bool complete_counted = false;
  bool suspend = false;
  std::uint64_t task_id = 0u;
};

class ChannelAccess final {
public:
  ChannelAccess() = delete;

  [[nodiscard]] static ActiveState ActiveSchedulerState() noexcept;
  [[nodiscard]] static std::size_t TaskSlot(std::uint64_t task_id) noexcept;
  [[nodiscard]] static ChannelDecision ParkChannelWait(std::uint64_t channel_id,
                                                       bool send_wait) noexcept;
  [[nodiscard]] static ChannelDecision
  WakeChannelTask(std::uint64_t task_id, std::uint64_t channel_id) noexcept;
  [[nodiscard]] static ChannelDecision
  WakeChannelWaiters(std::vector<std::uint64_t> task_ids,
                     std::uint64_t channel_id,
                     std::vector<std::uint64_t> *failed_out) noexcept;
  [[nodiscard]] static ChannelDecision
  MakeChannelRecord(std::size_t capacity,
                    std::uint64_t *out_channel_id) noexcept;
  [[nodiscard]] static ChannelDecision
  ReleaseChannelRecord(std::uint64_t channel_id, std::size_t capacity) noexcept;
  static void ReleaseCommittedChannelRecord(std::uint64_t channel_id,
                                            std::size_t capacity) noexcept;
  [[nodiscard]] static ChannelDecision
  RecordChannelSend(std::uint64_t channel_id,
                    std::uint64_t value_count) noexcept;
  [[nodiscard]] static ChannelDecision
  RecordBufferedChannelSendBatch(std::uint64_t channel_id,
                                 std::uint64_t value_count,
                                 std::uint64_t logical_sends) noexcept;
  [[nodiscard]] static ChannelDecision
  RecordChannelRecv(std::uint64_t channel_id,
                    std::uint64_t value_count) noexcept;
  [[nodiscard]] static ChannelDecision
  RecordChannelClose(std::uint64_t channel_id) noexcept;
  static void RecordCommittedChannelClose(std::uint64_t channel_id) noexcept;
  [[nodiscard]] static ChannelDecision CommitSchedulerOperation(
      OperationKind operation_kind = OperationKind::PrimitiveTrap) noexcept;
  [[nodiscard]] static ChannelDecision
  CommitSchedulerOperation(OperationKind operation_kind,
                           ActiveState *active_state) noexcept;
  [[nodiscard]] static ChannelDecision
  CommitSchedulerOperationLight(OperationKind operation_kind,
                                std::uint64_t *scheduler_id,
                                std::uint64_t *task_id) noexcept;
  [[nodiscard]] static ChannelDecision FinishCurrentOperation() noexcept;
  [[nodiscard]] static ChannelDecision FinishCommittedOperation() noexcept;
  [[nodiscard]] static ChannelDecision FinishCommit(bool counted) noexcept;
};

} // namespace rund::detail::task
