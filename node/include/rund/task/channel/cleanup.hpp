void CompleteWokenWaiter(const std::uint64_t task_id) noexcept {
  CompleteWokenWaiters(&task_id, 1u);
}

void CompleteWokenWaiters(const std::vector<std::uint64_t> &task_ids) noexcept {
  CompleteWokenWaiters(task_ids.data(), task_ids.size());
}

void CompleteWokenWaiters(const std::uint64_t *const task_ids,
                          const std::size_t count) noexcept {
  if (control_ == nullptr) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(control_->mutex);
    for (std::size_t index = 0u; index < count; ++index) {
      CompleteCountedWake(task_ids[index]);
    }
  }
  TryReleaseSchedulerRecord();
}

void RemoveCompletedSend(const std::uint64_t task_id) noexcept {
  const std::size_t index = TaskSlot(task_id);
  if (index >= control_->completed_send_by_task.size()) {
    return;
  }
  std::optional<CompletedSend> &slot = control_->completed_send_by_task[index];
  if (!slot) {
    return;
  }
  slot.reset();
  if (control_->completed_send_count > 0u) {
    --control_->completed_send_count;
  }
}

void RemoveRendezvous(const std::uint64_t receiver_task_id) noexcept {
  const std::size_t index = TaskSlot(receiver_task_id);
  if (index >= control_->rendezvous_by_receiver.size()) {
    return;
  }
  std::optional<Rendezvous> &slot = control_->rendezvous_by_receiver[index];
  if (!slot) {
    return;
  }
  slot.reset();
  if (control_->rendezvous_count > 0u) {
    --control_->rendezvous_count;
  }
}

void CompleteCountedWake(const std::uint64_t task_id) noexcept {
  const std::size_t index = TaskSlot(task_id);
  if (index >= control_->counted_wake_count.size()) {
    return;
  }
  std::uint32_t &count = control_->counted_wake_count[index];
  if (count == 0u) {
    return;
  }
  --count;
  if (control_->counted_wake_entries > 0u) {
    --control_->counted_wake_entries;
  }
  if (control_->in_flight_wakes > 0u) {
    --control_->in_flight_wakes;
  }
}

void FailCloseAfterWakeFailure(const std::uint64_t failed_task_id) noexcept {
  if (control_ == nullptr) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(control_->mutex);
    control_->closed = true;
    ClearBuffered();
    control_->send_waiter_head = 0u;
    control_->send_waiter_tail = 0u;
    control_->recv_waiter_head = 0u;
    control_->recv_waiter_tail = 0u;
    control_->send_waiting.clear();
    control_->recv_waiting.clear();
    control_->next_send_waiter.clear();
    control_->next_recv_waiter.clear();
    control_->send_value_by_task.clear();
    control_->send_waiter_count = 0u;
    control_->recv_waiter_count = 0u;
    RemoveRendezvous(failed_task_id);
    RemoveCompletedSend(failed_task_id);
    CompleteCountedWake(failed_task_id);
  }
  TryReleaseSchedulerRecord();
}

void TryReleaseSchedulerRecord() noexcept {
  if (control_ == nullptr) {
    return;
  }
  std::uint64_t channel_id = 0u;
  std::size_t capacity = 0u;
  bool should_release = false;
  {
    std::lock_guard<std::mutex> lock(control_->mutex);
    if (!control_->scheduler_released && control_->closed && BufferEmpty() &&
        control_->rendezvous_count == 0u && control_->send_waiter_count == 0u &&
        control_->completed_send_count == 0u &&
        control_->recv_waiter_count == 0u &&
        control_->counted_wake_entries == 0u &&
        control_->in_flight_wakes == 0u) {
      control_->scheduler_released = true;
      channel_id = control_->channel_id;
      capacity = control_->capacity;
      should_release = true;
    }
  }
  if (should_release) {
    ::rund::detail::task::ChannelAccess::ReleaseCommittedChannelRecord(
        channel_id, capacity);
  }
}

void Abandon() noexcept {
  if (!owns_ || control_ == nullptr) {
    return;
  }
  owns_ = false;
  bool closed = false;
  {
    std::lock_guard<std::mutex> lock(control_->mutex);
    closed = control_->closed;
  }
  if (!closed) {
    (void)close();
  }
  {
    std::lock_guard<std::mutex> lock(control_->mutex);
    ClearBuffered();
  }
  TryReleaseSchedulerRecord();
}
