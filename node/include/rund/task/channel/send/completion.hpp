  void CompleteParkedSend(const std::uint64_t task_id,
                          const std::uint64_t value_count) {
    const std::size_t index = TaskSlot(task_id);
    if (index >= control_->completed_send_by_task.size()) return;
    std::optional<CompletedSend>& slot =
        control_->completed_send_by_task[index];
    if (!slot) {
      ++control_->completed_send_count;
    }
    slot.emplace(CompletedSend{.task_id = task_id, .value_count = value_count});
    AddCountedWake(task_id);
  }

  void AddCountedWake(const std::uint64_t task_id) {
    const std::size_t index = TaskSlot(task_id);
    if (index >= control_->counted_wake_count.size()) return;
    ++control_->counted_wake_count[index];
    ++control_->counted_wake_entries;
    ++control_->in_flight_wakes;
  }

  [[nodiscard]] bool TakeCompletedSend(
      const std::uint64_t task_id,
      std::uint64_t* const value_count) noexcept {
    const std::size_t index = TaskSlot(task_id);
    if (index >= control_->completed_send_by_task.size()) {
      return false;
    }
    std::optional<CompletedSend>& slot =
        control_->completed_send_by_task[index];
    if (!slot) {
      return false;
    }
    if (value_count != nullptr) {
      *value_count = slot->value_count;
    }
    slot.reset();
    if (control_->completed_send_count > 0u) {
      --control_->completed_send_count;
    }
    return true;
  }
