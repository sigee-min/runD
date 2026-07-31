[[nodiscard]] ::rund::detail::task::ChannelDecision
ParkChannel(std::uint64_t channel_id, bool send_wait) noexcept;
[[nodiscard]] ::rund::detail::task::ChannelDecision
WakeChannel(std::uint64_t task_id, std::uint64_t channel_id) noexcept;
[[nodiscard]] ::rund::detail::task::ChannelDecision
ReleaseChannel(std::uint64_t channel_id, std::size_t capacity) noexcept;
void ReleaseCommittedChannel(std::uint64_t channel_id,
                             std::size_t capacity) noexcept;
[[nodiscard]] ::rund::detail::task::ChannelDecision
MakeChannel(std::size_t capacity, std::uint64_t *out_channel_id) noexcept;
[[nodiscard]] ::rund::detail::task::ChannelDecision
RecordChannel(::rund::detail::task::OperationKind kind,
              std::uint64_t channel_id, std::uint64_t value_count) noexcept;
[[nodiscard]] ::rund::detail::task::ChannelDecision
RecordBufferedChannelSendBatch(std::uint64_t channel_id,
                               std::uint64_t value_count,
                               std::uint64_t logical_sends) noexcept;
void RecordCommittedChannel(::rund::detail::task::OperationKind kind,
                            std::uint64_t channel_id,
                            std::uint64_t value_count) noexcept;
