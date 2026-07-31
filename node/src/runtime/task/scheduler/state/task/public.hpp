[[nodiscard]] task::Handle
Spawn(const char *name, ::rund::detail::task::Callable callable) noexcept;
[[nodiscard]] task::Handle Spawn(const char *name,
                                 task::Task<void> &&task) noexcept;
[[nodiscard]] ::rund::detail::task::Spawned
SpawnObserved(const char *name,
              ::rund::detail::task::CoroutineStart start) noexcept;
[[nodiscard]] ::rund::detail::task::Spawned
SpawnAwaited(::rund::detail::task::CoroutineStart start) noexcept;
[[nodiscard]] bool CurrentTaskIsCoroutine() const noexcept;
[[nodiscard]] bool ParkExternal(std::atomic<std::uint8_t> &phase,
                                ExternalWake &wake) noexcept;
[[nodiscard]] bool WakeExternal(ExternalWake wake) noexcept;
[[nodiscard]] task::Handle CurrentHandle() noexcept;
[[nodiscard]] bool DispatchExternal(const task::Handle &handle) noexcept;
[[nodiscard]] task::Status Join(const task::Handle *handles,
                                std::size_t count) noexcept;
[[nodiscard]] ::rund::detail::task::AwaitDecision
BeginJoinAwait(const task::Handle *handles, std::size_t count) noexcept;
[[nodiscard]] task::Status Drain() noexcept;
[[nodiscard]] ::rund::detail::task::AwaitDecision Yield() noexcept;
[[nodiscard]] ::rund::detail::task::AwaitDecision
Sleep(std::chrono::nanoseconds duration) noexcept;
