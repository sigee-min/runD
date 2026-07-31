struct OperationGateActive {
  bool active = false;
  std::uint64_t scheduler_id = 0u;
  std::uint64_t task_id = 0u;

  OperationGateActive() noexcept = default;

  OperationGateActive(::rund::detail::task::ActiveState state) noexcept
      : active(state.active), scheduler_id(state.scheduler_id),
        task_id(state.task_id) {}
};

struct OperationGate {
  ::rund::detail::task::ChannelDecision result{};
  OperationGateActive active{};
};
