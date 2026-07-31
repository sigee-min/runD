  [[nodiscard]] task::Status CreateStopSource(
      std::uint64_t* scheduler_id,
      std::uint64_t* source_id,
      std::uint64_t* generation,
      std::uint64_t* epoch) noexcept;
  [[nodiscard]] task::Status RequestStop(
      std::uint64_t scheduler_id,
      std::uint64_t source_id,
      std::uint64_t generation,
      std::uint64_t epoch) noexcept;
  [[nodiscard]] task::StopState StopRequested(
      std::uint64_t scheduler_id,
      std::uint64_t source_id,
      std::uint64_t generation,
      std::uint64_t epoch) noexcept;
  [[nodiscard]] task::StopState StopRequestedUnsequenced(
      std::uint64_t scheduler_id,
      std::uint64_t source_id,
      std::uint64_t generation,
      std::uint64_t epoch) const noexcept;
