[[nodiscard]] ::rund::detail::task::IoDecision
WaitReactor(int fd, short interest, std::uint64_t host_handle_id = 0u,
            std::uint64_t fd_generation = 0u,
            ::rund::net::SocketView socket = {}) noexcept;
[[nodiscard]] ::rund::net::CloseResult
CloseSocket(::rund::net::SocketView socket, int native_socket) noexcept;
[[nodiscard]] ::rund::detail::task::IoDecision WaitReactorTimed(
    int fd, short interest, std::chrono::nanoseconds timeout,
    std::uint64_t host_handle_id = 0u, std::uint64_t fd_generation = 0u,
    std::uint64_t stop_scheduler_id = 0u, std::uint64_t stop_source_id = 0u,
    std::uint64_t stop_generation = 0u, std::uint64_t stop_epoch = 0u,
    ::rund::net::SocketView socket = {}) noexcept;
[[nodiscard]] ::rund::net::ready::many::Wait
WaitReactorMany(std::span<const ::rund::net::ready::Request> requests,
                std::span<::rund::net::ready::Event> out,
                std::optional<std::chrono::nanoseconds> timeout,
                ::rund::net::ready::many::Budget budget,
                std::uint64_t stop_scheduler_id = 0u,
                std::uint64_t stop_source_id = 0u,
                std::uint64_t stop_generation = 0u,
                std::uint64_t stop_epoch = 0u, std::uint64_t ready_set_id = 0u,
                std::uint64_t ready_set_generation = 0u) noexcept;
[[nodiscard]] ::rund::net::ready::many::Wait
ResumeReactorMany(std::span<::rund::net::ready::Event> out,
                  std::uint64_t group_id) noexcept;
[[nodiscard]] ::rund::net::ready::Status
CreateReadySet(::rund::net::ready::Config options) noexcept;
[[nodiscard]] ::rund::net::ready::Status
DestroyReadySet(::rund::net::ready::Set set) noexcept;
[[nodiscard]] ::rund::net::ready::Status
ClearReadySet(::rund::net::ready::Set set) noexcept;
[[nodiscard]] ::rund::net::ready::Status
AddReadyInterest(::rund::net::ready::Set set,
                 ::rund::net::ready::Request request) noexcept;
[[nodiscard]] ::rund::net::ready::Status
RemoveReadyInterest(::rund::net::ready::Set set,
                    ::rund::net::ready::Request request) noexcept;
[[nodiscard]] ::rund::net::Limits ReadLimits() noexcept;
[[nodiscard]] ::rund::net::ready::many::Wait
WaitReadySet(::rund::net::ready::Set set,
             std::span<::rund::net::ready::Event> out,
             std::optional<std::chrono::nanoseconds> timeout,
             ::rund::net::ready::many::Budget budget) noexcept;
