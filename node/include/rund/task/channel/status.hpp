template <typename U> struct RecvDecision final {
  ::rund::detail::task::ChannelDecision decision{};
  std::optional<U> value{};
};
