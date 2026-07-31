struct ReplayInputCaptureState final {
  bool mutated = false;
  std::uint64_t token = 0u;
  replay_detail::payload::InputBinding binding{};
  std::uint64_t event_offset = 0u;
  std::uint64_t event_sequence = 0u;
  std::uint64_t payload_offset = 0u;
  std::uint64_t operation_count = 0u;
  std::uint64_t simulation_fingerprint = 0u;
  std::size_t byte_offset = 0u;
};

struct SchedulerEvidenceState {
  mutable std::recursive_mutex mutex{};
  std::vector<task::Observation> observations{};
  std::vector<::rund::host::Event> host_events{};
  replay_detail::payload::Store host_payload_store{};
  std::uint64_t host_payload_reserved_bytes = 0u;
  std::shared_ptr<std::vector<std::byte>> input_bytes{};
  std::size_t input_byte_size = 0u;
  std::uint64_t input_consumed_bytes = 0u;
  std::uint32_t input_capacity = 0u;
  std::uint32_t input_count = 0u;
  std::atomic<bool> input_capture_active{false};
  ReplayInputCaptureState input_capture{};
  std::uint64_t next_input_capture_token = 1u;
  ::rund::detail::task::StatStorage metrics{};
};
