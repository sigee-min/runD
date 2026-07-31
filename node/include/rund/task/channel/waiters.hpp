[[nodiscard]] std::size_t TaskSlot(const std::uint64_t task_id) const noexcept {
  return ::rund::detail::task::ChannelAccess::TaskSlot(task_id);
}

[[nodiscard]] bool
SendWaiterActive(const std::uint64_t task_id) const noexcept {
  const std::size_t slot = TaskSlot(task_id);
  return slot < control_->send_waiting.size() &&
         control_->send_waiting[slot] != 0u;
}

[[nodiscard]] bool
RecvWaiterActive(const std::uint64_t task_id) const noexcept {
  const std::size_t slot = TaskSlot(task_id);
  return slot < control_->recv_waiting.size() &&
         control_->recv_waiting[slot] != 0u;
}

[[nodiscard]] bool HasSendWaiters() const noexcept {
  return control_->send_waiter_count != 0u;
}

[[nodiscard]] bool HasRecvWaiters() const noexcept {
  return control_->recv_waiter_count != 0u;
}
