[[nodiscard]] bool RecordHostEvent(::rund::host::Event event) noexcept;
[[nodiscard]] bool CapturesNetIngress() const noexcept;
[[nodiscard]] bool
RecordHostEvents(const std::vector<::rund::host::Event> &events) noexcept;
[[nodiscard]] bool
RecordHostEventFromHostApi(::rund::host::Event event) noexcept;
[[nodiscard]] bool RecordHostEventFromHostApi(
    ::rund::host::Event event,
    const replay_detail::payload::RawByteSource &source) noexcept;
[[nodiscard]] ::rund::host::io::CloseResult
CloseHostFd(::rund::host::io::FdView fd) noexcept;
[[nodiscard]] scheduler_host::ReplayInputMode ReplayInputMode() const noexcept;
[[nodiscard]] scheduler_host::ReplayInputCapture
BeginReplayInput(const replay_detail::payload::InputBinding &binding) noexcept;
void FailReplayInput(::rund::replay::Code code) noexcept;
void CancelReplayInput(scheduler_host::ReplayInputCapture capture) noexcept;
[[nodiscard]] replay_detail::payload::ResolveResult
RejectReplayInput(scheduler_host::ReplayInputCapture capture,
                  ::rund::replay::Code code) noexcept;
[[nodiscard]] replay_detail::payload::ResolveResult
FinishReplayInput(const replay_detail::payload::InputBinding &binding,
                  scheduler_host::ReplayInputCapture capture,
                  std::size_t byte_count) noexcept;
[[nodiscard]] replay_detail::payload::ResolveResult
ReplayInput(const replay_detail::payload::InputBinding &binding) noexcept;
