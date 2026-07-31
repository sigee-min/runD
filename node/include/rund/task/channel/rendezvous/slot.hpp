  void PushRendezvous(const std::uint64_t receiver_task_id, T&& value) {
    const std::size_t index = TaskSlot(receiver_task_id);
    if (index >= control_->rendezvous_by_receiver.size()) return;
    std::optional<Rendezvous>& slot =
        control_->rendezvous_by_receiver[index];
    if (!slot) {
      ++control_->rendezvous_count;
    }
    slot.emplace(Rendezvous{receiver_task_id, std::move(value)});
  }

  [[nodiscard]] bool TakeRendezvous(
      const std::uint64_t receiver_task_id,
      std::optional<T>* const value) {
    const std::size_t index = TaskSlot(receiver_task_id);
    if (index >= control_->rendezvous_by_receiver.size()) {
      return false;
    }
    std::optional<Rendezvous>& slot =
        control_->rendezvous_by_receiver[index];
    if (!slot) {
      return false;
    }
    if (value != nullptr) {
      value->emplace(std::move(slot->value));
    }
    slot.reset();
    if (control_->rendezvous_count > 0u) {
      --control_->rendezvous_count;
    }
    return true;
  }
