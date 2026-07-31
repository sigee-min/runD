  struct PendingSend {
    std::uint64_t task_id = 0u;
    std::optional<T> value{};
  };
