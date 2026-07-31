struct SchedulerReactorState {
  SchedulerReactorState() noexcept;
  ~SchedulerReactorState();

  ReactorRuntime reactor{};
  std::vector<ReactorManyGroup> reactor_many_groups{};
  std::vector<ReactorManyRequest> reactor_many_requests{};
  std::vector<ReactorManyRequest> reactor_many_request_scratch{};
  std::vector<std::uint32_t> reactor_many_index_scratch{};
  std::vector<std::uint64_t> reactor_many_group_id_scratch{};
  std::vector<ReactorManyEventSlot> reactor_many_event_slots{};
  std::vector<ReactorManyEventSlot> reactor_many_event_slots_scratch{};
  std::vector<BatchIoPollRequest> reactor_many_poll_request_scratch{};
  std::vector<BatchIoReady> reactor_many_ready_result_scratch{};
  std::vector<ReasonCode> reactor_ready_code_scratch{};
  std::vector<::rund::net::SocketLease> reactor_socket_lease_scratch{};
  std::vector<ReactorReadySet> reactor_ready_sets{};
  std::uint64_t reactor_many_validation_comparisons = 0u;
  std::uint64_t reactor_many_request_copies = 0u;
  std::uint64_t reactor_many_storage_growths = 0u;
  std::uint64_t reactor_ready_set_storage_growths = 0u;
  std::shared_ptr<std::atomic<std::uint32_t>>
      live_net_socket_registry_entries{};
  std::vector<StopSourceRecord> stop_sources{};
  std::vector<ReactorWait> canceled_wait_scratch{};
  std::vector<::rund::host::Event> reactor_host_event_scratch{};
};
