struct ScopeToken {
  std::uint64_t scope_id = 0u;
  std::uint64_t previous_scope_id = 0u;
  std::size_t observation_begin = 0u;
  std::size_t event_begin = 0u;
  ReasonCode code = ReasonCode::TaskInvalid;
};
