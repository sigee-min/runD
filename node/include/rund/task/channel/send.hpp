class SendOp {
public:
  class Awaiter {
  public:
    Awaiter(std::shared_ptr<Control> control, std::optional<T> value,
            const ReasonCode code) noexcept
        : control_(std::move(control)), value_(std::move(value)), code_(code) {}

    [[nodiscard]] bool await_ready() const noexcept { return false; }

    bool await_suspend(std::coroutine_handle<>) {
      channel view{};
      view.control_ = control_;
      view.code_ = code_;
      if (!value_) {
        decision_.status = Status::fail(ReasonCode::ChannelInvalid);
        return false;
      }
      decision_ = view.send_impl(std::move(*value_));
      value_.reset();
      return decision_.status && decision_.suspend;
    }

    [[nodiscard]] Status await_resume() {
      channel view{};
      view.control_ = control_;
      ::rund::detail::task::ChannelDecision decision =
          decision_.suspend ? CoroutineResumeSend(control_, decision_.task_id)
                            : std::move(decision_);
      return view.PublicStatus(std::move(decision));
    }

  private:
    std::shared_ptr<Control> control_{};
    std::optional<T> value_{};
    ReasonCode code_ = ReasonCode::ChannelInvalid;
    ::rund::detail::task::ChannelDecision decision_{};
  };

  SendOp(
      std::shared_ptr<Control> control, T value,
      const ReasonCode code) noexcept(std::is_nothrow_move_constructible_v<T>)
      : control_(std::move(control)), value_(std::move(value)), code_(code) {}

  SendOp(const SendOp &) = delete;
  SendOp &operator=(const SendOp &) = delete;
  SendOp(SendOp &&) noexcept = default;
  SendOp &operator=(SendOp &&) noexcept = default;

  [[nodiscard]] Awaiter operator co_await() && noexcept {
    return Awaiter{std::move(control_), std::move(value_), code_};
  }

private:
  std::shared_ptr<Control> control_{};
  std::optional<T> value_{};
  ReasonCode code_ = ReasonCode::ChannelInvalid;
};

[[nodiscard]] ::rund::detail::task::ChannelDecision
send_impl(T value) noexcept(std::is_nothrow_move_constructible_v<T>) {
  const OperationGate gate = BeginChannelOperation(
      true, ::rund::detail::task::OperationKind::ChannelSend);
  if (!gate.result.status) {
    return Finalize(gate.result);
  }
  const std::uint64_t task_id = gate.active.task_id;
  bool queued = false;
  for (;;) {
    std::uint64_t wake_receiver = 0u;
    std::uint64_t record_value_count = 0u;
    bool completed_send = false;
    {
      std::lock_guard<std::mutex> lock(control_->mutex);
      if (queued && TakeCompletedSend(task_id, &record_value_count)) {
        completed_send = true;
      } else if (control_->closed) {
        return FinalizeCommitted(::rund::detail::task::ChannelDecision{
            .status = Status::fail(ReasonCode::ChannelClosed)});
      } else if (!queued && HasRecvWaiters() &&
                 (control_->capacity == 0u || !BufferFull())) {
        if (!EnsureRendezvousStorage()) {
          return FinalizeCommitted(::rund::detail::task::ChannelDecision{
              .status = Status::fail(ReasonCode::ChannelCapacityExceeded)});
        }
        if (PopRecvWaiter(&wake_receiver)) {
          PushRendezvous(wake_receiver, std::move(value));
        }
      } else if (!queued && !BufferFull()) {
        PushBuffered(std::move(value));
      } else if (!queued) {
        if (!PushSendWaiter(task_id, std::move(value))) {
          return FinalizeCommitted(::rund::detail::task::ChannelDecision{
              .status = Status::fail(ReasonCode::ChannelCapacityExceeded)});
        }
        queued = true;
      } else if (!SendWaiterActive(task_id)) {
        record_value_count = BufferSize();
        return FinalizeCommitted(
            ::rund::detail::task::ChannelAccess::RecordChannelSend(
                control_->channel_id, record_value_count),
            true);
      }
      if (!completed_send) {
        record_value_count = BufferSize();
      }
    }
    if (completed_send) {
      ::rund::detail::task::ChannelDecision record =
          ::rund::detail::task::ChannelAccess::RecordChannelSend(
              control_->channel_id, record_value_count);
      TryReleaseSchedulerRecord();
      return FinalizeCommitted(record, true);
    }
    if (wake_receiver != 0u) {
      const ::rund::detail::task::ChannelDecision wake =
          ::rund::detail::task::ChannelAccess::WakeChannelTask(
              wake_receiver, control_->channel_id);
      if (!wake.status) {
        FailCloseAfterWakeFailure(wake_receiver);
        return FinalizeCommitted(wake);
      }
      return FinalizeCommitted(
          ::rund::detail::task::ChannelAccess::RecordChannelSend(
              control_->channel_id, record_value_count),
          true);
    }
    if (!queued) {
      return FinalizeCommitted(
          ::rund::detail::task::ChannelAccess::RecordChannelSend(
              control_->channel_id, record_value_count),
          true);
    }
    ::rund::detail::task::ChannelDecision parked =
        ::rund::detail::task::ChannelAccess::ParkChannelWait(
            control_->channel_id, true);
    if (parked.suspend) {
      parked.task_id = task_id;
      return parked;
    }
    const ::rund::detail::task::ChannelDecision committed =
        ::rund::detail::task::ChannelAccess::CommitSchedulerOperation(
            ::rund::detail::task::OperationKind::ChannelSend);
    CompleteWokenWaiter(task_id);
    if (!committed.status) {
      return FinalizeCommitted(committed);
    }
    if (!parked.status) {
      std::lock_guard<std::mutex> lock(control_->mutex);
      RemoveSendWaiter(task_id);
      return FinalizeCommitted(parked);
    }
  }
}

[[nodiscard]] SendOp
send(T value) noexcept(std::is_nothrow_move_constructible_v<T>) {
  return SendOp{control_, std::move(value), code_};
}
