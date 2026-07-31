class RecvOp {
public:
  class Awaiter {
  public:
    Awaiter(std::shared_ptr<Control> control, const ReasonCode code) noexcept
        : control_(std::move(control)), code_(code) {}

    [[nodiscard]] bool await_ready() const noexcept { return false; }

    bool await_suspend(std::coroutine_handle<>) {
      channel view{};
      view.control_ = control_;
      view.code_ = code_;
      result_ = view.recv_impl();
      return result_.decision.status && result_.decision.suspend;
    }

    [[nodiscard]] ReceiveResult<T> await_resume() {
      channel view{};
      view.control_ = control_;
      RecvDecision<T> result =
          result_.decision.suspend
              ? CoroutineResumeRecv(control_, result_.decision.task_id)
              : std::move(result_);
      return view.PublicRecv(std::move(result));
    }

  private:
    std::shared_ptr<Control> control_{};
    ReasonCode code_ = ReasonCode::ChannelInvalid;
    RecvDecision<T> result_{};
  };

  RecvOp(std::shared_ptr<Control> control, const ReasonCode code) noexcept
      : control_(std::move(control)), code_(code) {}

  RecvOp(const RecvOp &) = delete;
  RecvOp &operator=(const RecvOp &) = delete;
  RecvOp(RecvOp &&) noexcept = default;
  RecvOp &operator=(RecvOp &&) noexcept = default;

  [[nodiscard]] Awaiter operator co_await() && noexcept {
    return Awaiter{std::move(control_), code_};
  }

private:
  std::shared_ptr<Control> control_{};
  ReasonCode code_ = ReasonCode::ChannelInvalid;
};

[[nodiscard]] RecvDecision<T>
recv_impl() noexcept(std::is_nothrow_move_constructible_v<T>) {
  const OperationGate gate = BeginChannelOperation(
      true, ::rund::detail::task::OperationKind::ChannelRecv);
  if (!gate.result.status) {
    return FinalizeRecv(RecvDecision<T>{.decision = gate.result});
  }
  const std::uint64_t task_id = gate.active.task_id;
  bool queued = false;
  for (;;) {
    std::optional<T> value{};
    bool has_value = false;
    std::uint64_t wake_sender = 0u;
    std::uint64_t wake_receiver = 0u;
    std::uint64_t record_value_count = 0u;
    {
      std::lock_guard<std::mutex> lock(control_->mutex);
      if (HasRecvWaiters() && !BufferEmpty() && !EnsureRendezvousStorage()) {
        return FinalizeRecv(RecvDecision<T>{
            .decision = ::rund::detail::task::ChannelDecision{
                .status = Status::fail(ReasonCode::ChannelCapacityExceeded),
                .complete_committed = true}});
      }
      if (TakeRendezvous(task_id, &value)) {
        has_value = true;
        if (!BufferEmpty() && HasRecvWaiters() &&
            PopRecvWaiter(&wake_receiver)) {
          PushRendezvous(wake_receiver, PopBuffered());
          AddCountedWake(wake_receiver);
        }
      } else if (!BufferEmpty()) {
        value.emplace(PopBuffered());
        has_value = true;
        PendingSend pending{};
        if (PopSendWaiter(&pending)) {
          PushBuffered(std::move(*pending.value));
          CompleteParkedSend(pending.task_id, BufferSize());
          wake_sender = pending.task_id;
        }
        if (!BufferEmpty() && HasRecvWaiters() &&
            PopRecvWaiter(&wake_receiver)) {
          PushRendezvous(wake_receiver, PopBuffered());
          AddCountedWake(wake_receiver);
        }
      } else if (HasSendWaiters()) {
        PendingSend pending{};
        (void)PopSendWaiter(&pending);
        value.emplace(std::move(*pending.value));
        CompleteParkedSend(pending.task_id, BufferSize());
        wake_sender = pending.task_id;
        has_value = true;
      } else if (control_->closed) {
        return FinalizeRecv(RecvDecision<T>{
            .decision = ::rund::detail::task::ChannelDecision{
                .status = Status::fail(ReasonCode::ChannelClosed),
                .complete_committed = true}});
      } else if (!queued) {
        if (!PushRecvWaiter(task_id)) {
          return FinalizeRecv(RecvDecision<T>{
              .decision = ::rund::detail::task::ChannelDecision{
                  .status = Status::fail(ReasonCode::ChannelCapacityExceeded),
                  .complete_committed = true}});
        }
        queued = true;
      }
      record_value_count = BufferSize();
    }
    if (wake_sender != 0u) {
      const ::rund::detail::task::ChannelDecision wake =
          ::rund::detail::task::ChannelAccess::WakeChannelTask(
              wake_sender, control_->channel_id);
      if (!wake.status) {
        FailCloseAfterWakeFailure(wake_sender);
        ::rund::detail::task::ChannelDecision failure = wake;
        failure.complete_committed = true;
        return FinalizeRecv(RecvDecision<T>{.decision = std::move(failure)});
      }
    }
    if (wake_receiver != 0u) {
      const ::rund::detail::task::ChannelDecision wake =
          ::rund::detail::task::ChannelAccess::WakeChannelTask(
              wake_receiver, control_->channel_id);
      if (!wake.status) {
        FailCloseAfterWakeFailure(wake_receiver);
        ::rund::detail::task::ChannelDecision failure = wake;
        failure.complete_committed = true;
        return FinalizeRecv(RecvDecision<T>{.decision = std::move(failure)});
      }
    }
    if (has_value) {
      ::rund::detail::task::ChannelDecision record =
          ::rund::detail::task::ChannelAccess::RecordChannelRecv(
              control_->channel_id, record_value_count);
      record.complete_committed = true;
      record.complete_counted = true;
      TryReleaseSchedulerRecord();
      return FinalizeRecv(RecvDecision<T>{.decision = std::move(record),
                                          .value = std::move(value)});
    }
    ::rund::detail::task::ChannelDecision parked =
        ::rund::detail::task::ChannelAccess::ParkChannelWait(
            control_->channel_id, false);
    if (parked.suspend) {
      parked.task_id = task_id;
      return RecvDecision<T>{.decision = std::move(parked)};
    }
    const ::rund::detail::task::ChannelDecision committed =
        ::rund::detail::task::ChannelAccess::CommitSchedulerOperation(
            ::rund::detail::task::OperationKind::ChannelRecv);
    CompleteWokenWaiter(task_id);
    if (!committed.status) {
      ::rund::detail::task::ChannelDecision failure = committed;
      failure.complete_committed = true;
      return FinalizeRecv(RecvDecision<T>{.decision = std::move(failure)});
    }
    if (!parked.status) {
      std::lock_guard<std::mutex> lock(control_->mutex);
      RemoveRecvWaiter(task_id);
      parked.complete_committed = true;
      return FinalizeRecv(RecvDecision<T>{.decision = std::move(parked)});
    }
  }
}

[[nodiscard]] RecvOp recv() noexcept(std::is_nothrow_move_constructible_v<T>) {
  return RecvOp{control_, code_};
}
