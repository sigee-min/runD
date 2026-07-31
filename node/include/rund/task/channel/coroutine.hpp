[[nodiscard]] static ::rund::detail::task::ChannelDecision
CoroutineResumeSend(const std::shared_ptr<Control> &context,
                    const std::uint64_t task_id) {
  channel view{};
  view.control_ = context;
  ::rund::detail::task::ChannelDecision resume =
      ::rund::detail::task::ChannelAccess::CommitSchedulerOperation(
          ::rund::detail::task::OperationKind::ChannelSend);
  view.CompleteWokenWaiter(task_id);
  if (!resume.status) {
    return resume;
  }
  std::uint64_t value_count = 0u;
  bool completed = false;
  bool closed = false;
  std::uint64_t channel_id = 0u;
  {
    std::lock_guard<std::mutex> lock(view.control_->mutex);
    channel_id = view.control_->channel_id;
    completed = view.TakeCompletedSend(task_id, &value_count);
    closed = view.control_->closed;
  }
  if (!completed) {
    return ::rund::detail::task::ChannelDecision{
        .status = Status::fail(closed ? ReasonCode::ChannelClosed
                                      : ReasonCode::TaskContextMissing),
        .complete_committed = true};
  }
  ::rund::detail::task::ChannelDecision record =
      ::rund::detail::task::ChannelAccess::RecordChannelSend(channel_id,
                                                             value_count);
  record.complete_committed = true;
  view.TryReleaseSchedulerRecord();
  return record;
}

[[nodiscard]] static RecvDecision<T>
CoroutineResumeRecv(const std::shared_ptr<Control> &context,
                    const std::uint64_t task_id) {
  channel view{};
  view.control_ = context;
  ::rund::detail::task::ChannelDecision resume =
      ::rund::detail::task::ChannelAccess::CommitSchedulerOperation(
          ::rund::detail::task::OperationKind::ChannelRecv);
  view.CompleteWokenWaiter(task_id);
  if (!resume.status) {
    return RecvDecision<T>{.decision = std::move(resume)};
  }
  std::optional<T> value{};
  bool has_value = false;
  bool closed = false;
  std::uint64_t channel_id = 0u;
  std::uint64_t value_count = 0u;
  {
    std::lock_guard<std::mutex> lock(view.control_->mutex);
    channel_id = view.control_->channel_id;
    has_value = view.TakeRendezvous(task_id, &value);
    closed = view.control_->closed;
    value_count = view.BufferSize();
  }
  if (!has_value) {
    return RecvDecision<T>{
        .decision = ::rund::detail::task::ChannelDecision{
            .status = Status::fail(closed ? ReasonCode::ChannelClosed
                                          : ReasonCode::TaskContextMissing),
            .complete_committed = true}};
  }
  ::rund::detail::task::ChannelDecision record =
      ::rund::detail::task::ChannelAccess::RecordChannelRecv(channel_id,
                                                             value_count);
  record.complete_committed = true;
  view.TryReleaseSchedulerRecord();
  return RecvDecision<T>{.decision = std::move(record),
                         .value = std::move(value)};
}
