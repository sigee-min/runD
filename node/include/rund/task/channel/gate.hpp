[[nodiscard]] ::rund::detail::task::ChannelDecision
EndpointGate() const noexcept {
  return !*this ? ::rund::detail::task::ChannelDecision{.status =
                                                            Status::fail(code_)}
                : ::rund::detail::task::ChannelDecision{.status =
                                                            Status::success()};
}

[[nodiscard]] OperationGate BeginChannelOperation(
    const bool require_task,
    const ::rund::detail::task::OperationKind operation_kind) const noexcept {
  ::rund::detail::task::ChannelDecision endpoint = EndpointGate();
  if (!endpoint.status) {
    return OperationGate{.result = endpoint};
  }
  std::uint64_t scheduler_id = 0u;
  std::uint64_t task_id = 0u;
  ::rund::detail::task::ChannelDecision commit =
      ::rund::detail::task::ChannelAccess::CommitSchedulerOperationLight(
          operation_kind, &scheduler_id, &task_id);
  if (!commit.status) {
    return OperationGate{.result = commit};
  }
  if (scheduler_id == 0u) {
    return OperationGate{
        .result = ::rund::detail::task::ChannelDecision{
            .status = Status::fail(ReasonCode::NodeRuntimeMissing),
            .complete_committed = true}};
  }
  const OperationGateActive active{::rund::detail::task::ActiveState{
      .active = true, .scheduler_id = scheduler_id, .task_id = task_id}};
  if (control_->owner_scheduler_id != scheduler_id) {
    return OperationGate{
        .result =
            ::rund::detail::task::ChannelDecision{
                .status = Status::fail(ReasonCode::ChannelWrongRuntime),
                .complete_committed = true},
        .active = active};
  }
  if (require_task && task_id == 0u) {
    return OperationGate{
        .result =
            ::rund::detail::task::ChannelDecision{
                .status = Status::fail(ReasonCode::TaskContextMissing),
                .complete_committed = true},
        .active = active};
  }
  return OperationGate{
      .result =
          ::rund::detail::task::ChannelDecision{.status = Status::success()},
      .active = active};
}
