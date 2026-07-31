[[nodiscard]] ::rund::detail::task::ActiveState ActiveState() noexcept;
[[nodiscard]] ScopeToken BeginScope() noexcept;
void FlushScopeBoundary() noexcept;
[[nodiscard]] task::Status EndScope(ScopeToken token) noexcept;
[[nodiscard]] bool
InstallPlan(const ::rund::replay::detail::scope::Plan &plan) noexcept;
void ClearPlan() noexcept;
[[nodiscard]] bool
RecordMemory(const ::rund::PreparedMemory &snapshot) noexcept;
[[nodiscard]] std::int64_t LogicalTimeNs() const noexcept;
[[nodiscard]] ::rund::host::random::RunSeed RandomSeed() const noexcept;
[[nodiscard]] ReactorInvariantSnapshot
ValidateReactorCleanupInvariants() noexcept;
[[nodiscard]] ScopeEvidence CaptureScope(std::size_t observation_begin,
                                         std::size_t event_begin);
[[nodiscard]] bool Ready() const noexcept;
[[nodiscard]] ReasonCode Code() const noexcept;
[[nodiscard]] const char *Reason() const noexcept;
void RefreshResourceStats() noexcept;
void RecordObservation(task::ObservationKind kind, ReasonCode code,
                       std::uint64_t task_id = 0u, std::uint64_t wait_id = 0u,
                       int fd = -1, short interest = 0, short revents = 0,
                       std::int64_t deadline_ns = 0) noexcept;
void RecordReactorObservation(task::ObservationKind kind, ReasonCode code,
                              std::uint64_t task_id, std::uint64_t wait_id,
                              int fd, short interest, short revents) noexcept;
[[nodiscard]] bool
RecordReactorHostEvent(ReasonCode code, std::uint64_t task_id,
                       std::uint64_t host_handle_id) noexcept;
[[nodiscard]] bool TryAdmitNetworkSocketRegistry() noexcept;
[[nodiscard]] ::rund::net::SocketRegistryOwner
ActiveNetworkSocketRegistryOwner() const noexcept;
void Reset() noexcept;
