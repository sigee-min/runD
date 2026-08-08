enum class HostEventPublication : std::uint8_t {
  Unpublished,
  Dropped,
  Retained,
};

class HostEventCommitResult final {
public:
  HostEventCommitResult() = delete;

  [[nodiscard]] static HostEventCommitResult
  unpublished_failure(const ReasonCode code) noexcept {
    if (code == ReasonCode::Ok) {
      std::abort();
    }
    return HostEventCommitResult{code, 0u, HostEventPublication::Unpublished};
  }

  [[nodiscard]] static HostEventCommitResult
  published(const ReasonCode code, const std::uint64_t sequence,
            const bool retained) noexcept {
    if (sequence == 0u) {
      std::abort();
    }
    return HostEventCommitResult{code, sequence,
                                 retained ? HostEventPublication::Retained
                                          : HostEventPublication::Dropped};
  }

  [[nodiscard]] constexpr ReasonCode code() const noexcept { return code_; }
  [[nodiscard]] constexpr bool ok() const noexcept {
    return code_ == ReasonCode::Ok;
  }
  [[nodiscard]] constexpr bool published() const noexcept {
    return publication_ != HostEventPublication::Unpublished;
  }
  [[nodiscard]] constexpr bool retained() const noexcept {
    return publication_ == HostEventPublication::Retained;
  }
  [[nodiscard]] constexpr std::uint64_t sequence() const noexcept {
    return sequence_;
  }

private:
  constexpr HostEventCommitResult(
      const ReasonCode code, const std::uint64_t sequence,
      const HostEventPublication publication) noexcept
      : sequence_(sequence), code_(code), publication_(publication) {}

  std::uint64_t sequence_;
  ReasonCode code_;
  HostEventPublication publication_;
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
