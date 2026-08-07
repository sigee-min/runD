  [[nodiscard]] ::rund::detail::task::StopIdentity CreateStopSource() noexcept;
  [[nodiscard]] task::Status RequestStop(
      ::rund::detail::task::StopIdentity identity) noexcept;
  [[nodiscard]] task::StopState StopRequested(
      ::rund::detail::task::StopIdentity identity) noexcept;
  [[nodiscard]] task::StopState StopRequestedUnsequenced(
      ::rund::detail::task::StopIdentity identity) const noexcept;
