[[nodiscard]] Status close() noexcept {
  const OperationGate gate = BeginChannelOperation(
      false, ::rund::detail::task::OperationKind::ChannelClose);
  if (!gate.result.status) {
    return PublicStatus(gate.result);
  }
  std::uint64_t channel_id = 0u;
  std::size_t released_capacity = 0u;
  bool release_now = false;
  std::vector<std::uint64_t> waiters{};
  {
    std::lock_guard<std::mutex> lock(control_->mutex);
    channel_id = control_->channel_id;
    released_capacity = control_->capacity;
    control_->closed = true;
    if (control_->send_waiter_count != 0u ||
        control_->recv_waiter_count != 0u) {
      PendingSend pending{};
      while (PopSendWaiter(&pending)) {
        waiters.push_back(pending.task_id);
        AddCountedWake(pending.task_id);
      }
      std::uint64_t waiter = 0u;
      while (PopRecvWaiter(&waiter)) {
        waiters.push_back(waiter);
        AddCountedWake(waiter);
      }
    }
    control_->send_waiter_head = 0u;
    control_->send_waiter_tail = 0u;
    control_->recv_waiter_head = 0u;
    control_->recv_waiter_tail = 0u;
    if (waiters.empty() && !control_->scheduler_released && BufferEmpty() &&
        control_->rendezvous_count == 0u && control_->send_waiter_count == 0u &&
        control_->completed_send_count == 0u &&
        control_->recv_waiter_count == 0u &&
        control_->counted_wake_entries == 0u &&
        control_->in_flight_wakes == 0u) {
      control_->scheduler_released = true;
      release_now = true;
    }
  }
  if (waiters.empty()) {
    ::rund::detail::task::ChannelAccess::RecordCommittedChannelClose(
        channel_id);
    if (release_now) {
      ::rund::detail::task::ChannelAccess::ReleaseCommittedChannelRecord(
          channel_id, released_capacity);
    }
    return FinishOk().status;
  }
  ::rund::detail::task::ChannelDecision close_record =
      ::rund::detail::task::ChannelAccess::RecordChannelClose(channel_id);
  std::vector<std::uint64_t> failed_wakes{};
  const ::rund::detail::task::ChannelDecision wake_result =
      ::rund::detail::task::ChannelAccess::WakeChannelWaiters(
          std::move(waiters), channel_id, &failed_wakes);
  CompleteWokenWaiters(failed_wakes);
  TryReleaseSchedulerRecord();
  return FinishOperation(wake_result.status ? close_record : wake_result)
      .status;
}
