[[nodiscard]] task::Status FailJoin(ReasonCode code) noexcept;
[[nodiscard]] task::Status FailScope(ReasonCode code) noexcept;
[[nodiscard]] ::rund::detail::task::AwaitDecision
FailYield(ReasonCode code) noexcept;
[[nodiscard]] ::rund::detail::task::AwaitDecision
FailSleep(ReasonCode code) noexcept;
[[nodiscard]] ::rund::detail::task::IoDecision
FailIo(ReasonCode code, short revents = 0) noexcept;
