[[nodiscard]] bool ValidateTimedReactorWait(
    int fd, std::chrono::nanoseconds timeout, std::uint64_t task_id,
    std::uint64_t host_handle_id, std::uint64_t fd_generation,
    ::rund::detail::task::StopIdentity stop,
    TaskRecord *&record, std::uint64_t &wait_host_handle_id,
    ::rund::detail::task::IoDecision &result) noexcept;
[[nodiscard]] bool ResolveImmediateTimedReactorWait(
    TaskRecord &record, int fd, short interest,
    std::chrono::nanoseconds timeout, std::uint64_t wait_host_handle_id,
    ::rund::detail::task::IoDecision &result) noexcept;
[[nodiscard]] bool
ParkTimedReactorWait(TaskRecord &record, int fd, short interest,
                     std::chrono::nanoseconds timeout,
                     std::uint64_t wait_host_handle_id,
                     std::uint64_t fd_generation,
                     ::rund::detail::task::StopSourceIdentity stop,
                     ::rund::net::SocketView socket, std::uint64_t &wait_id,
                     ::rund::detail::task::IoDecision &result) noexcept;
[[nodiscard]] ::rund::detail::task::IoDecision
ResumeTimedReactorWait(TaskRecord &record) noexcept;
