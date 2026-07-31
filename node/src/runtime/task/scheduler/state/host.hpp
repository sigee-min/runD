struct HostEventCommitResult {
  bool ok = false;
  bool retained = false;
  ReasonCode code = ReasonCode::HostReplayEventMismatch;
  std::uint64_t sequence = 0u;
  std::size_t expected_index = 0u;
  const char *reason = "host_replay_event_mismatch";
};

[[nodiscard]] ::rund::node::replay_detail::payload::Archive
CapturePayloads() const;
[[nodiscard]] HostEventCommitResult CommitHostEvent(
    ::rund::host::Event event,
    const replay_detail::payload::RawByteSource *source = nullptr) noexcept;
[[nodiscard]] ReasonCode RecordHostPayloadForCommittedEvent(
    const HostEventCommitResult &commit, ::rund::host::EventKind kind,
    const replay_detail::payload::Capture &payload) noexcept;
[[nodiscard]] bool ReserveHostPayloadCapacity(std::size_t bytes) noexcept;
void ReleaseHostPayloadCapacity(std::size_t bytes) noexcept;
void FailCurrentTaskOrScheduler(ReasonCode code) noexcept;
[[nodiscard]] bool
SuspendHostIoRead(::rund::host::io::FdView fd, std::span<std::byte> buffer,
                  void **token, ::rund::host::io::ReadResult *result) noexcept;
[[nodiscard]] bool
SuspendHostIoWrite(::rund::host::io::FdView fd,
                   std::span<const std::byte> buffer, void **token,
                   ::rund::host::io::WriteResult *result) noexcept;
[[nodiscard]] ::rund::host::io::ReadResult
CompleteHostIoRead(void *token) noexcept;
[[nodiscard]] ::rund::host::io::WriteResult
CompleteHostIoWrite(void *token) noexcept;
[[nodiscard]] bool PrepareHostIo() noexcept;
[[nodiscard]] bool SuspendHostIo(const HostIoOperation &operation, void **token,
                                 ReasonCode &failure) noexcept;
[[nodiscard]] HostIoCompletion CompleteHostIo(void *token,
                                              HostIoKind kind) noexcept;
[[nodiscard]] HostIoCompletion
ReplayHostIo(const HostIoOperation &operation) noexcept;
void MarkHostIoPayloadMismatch(::rund::replay::Code code) noexcept;
void StopHostIo() noexcept;
