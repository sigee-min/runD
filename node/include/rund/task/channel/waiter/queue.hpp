  [[nodiscard]] bool EnsureSendWaitStorage() noexcept {
    if (control_->send_waiting.size() == control_->task_capacity &&
        control_->next_send_waiter.size() == control_->task_capacity &&
        control_->send_value_by_task.size() == control_->task_capacity &&
        control_->completed_send_by_task.size() == control_->task_capacity &&
        control_->counted_wake_count.size() == control_->task_capacity) {
      return true;
    }
    try {
      control_->send_waiting.resize(control_->task_capacity, 0u);
      control_->next_send_waiter.resize(control_->task_capacity, 0u);
      control_->send_value_by_task.resize(control_->task_capacity);
      control_->completed_send_by_task.resize(control_->task_capacity);
      control_->counted_wake_count.resize(control_->task_capacity, 0u);
      return true;
    } catch (...) {
      return false;
    }
  }

  [[nodiscard]] bool EnsureRecvWaitStorage() noexcept {
    if (control_->recv_waiting.size() == control_->task_capacity &&
        control_->next_recv_waiter.size() == control_->task_capacity &&
        control_->counted_wake_count.size() == control_->task_capacity) {
      return true;
    }
    try {
      control_->recv_waiting.resize(control_->task_capacity, 0u);
      control_->next_recv_waiter.resize(control_->task_capacity, 0u);
      control_->counted_wake_count.resize(control_->task_capacity, 0u);
      return true;
    } catch (...) {
      return false;
    }
  }

  [[nodiscard]] bool EnsureRendezvousStorage() noexcept {
    if (control_->rendezvous_by_receiver.size() == control_->task_capacity) {
      return true;
    }
    try {
      control_->rendezvous_by_receiver.resize(control_->task_capacity);
      return true;
    } catch (...) {
      return false;
    }
  }

  [[nodiscard]] bool PushSendWaiter(const std::uint64_t task_id, T&& value) {
    if (!EnsureSendWaitStorage()) {
      return false;
    }
    const std::size_t index = TaskSlot(task_id);
    if (index >= control_->send_waiting.size()) {
      return false;
    }
    if (!SendWaiterActive(task_id)) {
      control_->send_waiting[index] = 1u;
      control_->next_send_waiter[index] = 0u;
      if (control_->send_waiter_tail != 0u) {
        const std::size_t tail = TaskSlot(control_->send_waiter_tail);
        if (tail < control_->next_send_waiter.size()) {
          control_->next_send_waiter[tail] = task_id;
        }
      } else {
        control_->send_waiter_head = task_id;
      }
      control_->send_waiter_tail = task_id;
      ++control_->send_waiter_count;
    }
    control_->send_value_by_task[index].emplace(std::move(value));
    return true;
  }

  [[nodiscard]] bool PushRecvWaiter(const std::uint64_t task_id) noexcept {
    if (!EnsureRecvWaitStorage()) {
      return false;
    }
    const std::size_t index = TaskSlot(task_id);
    if (index >= control_->recv_waiting.size()) {
      return false;
    }
    if (!RecvWaiterActive(task_id)) {
      control_->recv_waiting[index] = 1u;
      control_->next_recv_waiter[index] = 0u;
      if (control_->recv_waiter_tail != 0u) {
        const std::size_t tail = TaskSlot(control_->recv_waiter_tail);
        if (tail < control_->next_recv_waiter.size()) {
          control_->next_recv_waiter[tail] = task_id;
        }
      } else {
        control_->recv_waiter_head = task_id;
      }
      control_->recv_waiter_tail = task_id;
      ++control_->recv_waiter_count;
    }
    return true;
  }

  void RemoveSendWaiter(const std::uint64_t task_id) noexcept {
    if (!SendWaiterActive(task_id)) {
      return;
    }
    const std::size_t index = TaskSlot(task_id);
    if (index >= control_->send_waiting.size()) return;
    UnlinkSendWaiter(task_id);
    control_->send_waiting[index] = 0u;
    control_->next_send_waiter[index] = 0u;
    control_->send_value_by_task[index].reset();
    if (control_->send_waiter_count > 0u) {
      --control_->send_waiter_count;
    }
  }

  void RemoveRecvWaiter(const std::uint64_t task_id) noexcept {
    if (!RecvWaiterActive(task_id)) {
      return;
    }
    const std::size_t index = TaskSlot(task_id);
    if (index >= control_->recv_waiting.size()) return;
    UnlinkRecvWaiter(task_id);
    control_->recv_waiting[index] = 0u;
    control_->next_recv_waiter[index] = 0u;
    if (control_->recv_waiter_count > 0u) {
      --control_->recv_waiter_count;
    }
  }

  void UnlinkSendWaiter(const std::uint64_t task_id) noexcept {
    if (TaskSlot(task_id) >= control_->next_send_waiter.size()) {
      return;
    }
    std::uint64_t previous = 0u;
    std::uint64_t current = control_->send_waiter_head;
    const std::size_t limit = control_->next_send_waiter.size();
    for (std::size_t visited = 0u; current != 0u && visited < limit; ++visited) {
      const std::size_t current_index = TaskSlot(current);
      if (current_index >= control_->next_send_waiter.size()) {
        break;
      }
      std::uint64_t next = control_->next_send_waiter[current_index];
      if (next == current) {
        next = 0u;
      }
      if (current == task_id) {
        if (previous != 0u) {
          const std::size_t previous_index = TaskSlot(previous);
          if (previous_index < control_->next_send_waiter.size()) {
            control_->next_send_waiter[previous_index] = next;
          }
        } else {
          control_->send_waiter_head = next;
        }
        if (control_->send_waiter_tail == task_id) {
          control_->send_waiter_tail = previous;
        }
        control_->next_send_waiter[current_index] = 0u;
        return;
      }
      previous = current;
      current = next;
    }
  }

  void UnlinkRecvWaiter(const std::uint64_t task_id) noexcept {
    if (TaskSlot(task_id) >= control_->next_recv_waiter.size()) {
      return;
    }
    std::uint64_t previous = 0u;
    std::uint64_t current = control_->recv_waiter_head;
    const std::size_t limit = control_->next_recv_waiter.size();
    for (std::size_t visited = 0u; current != 0u && visited < limit; ++visited) {
      const std::size_t current_index = TaskSlot(current);
      if (current_index >= control_->next_recv_waiter.size()) {
        break;
      }
      std::uint64_t next = control_->next_recv_waiter[current_index];
      if (next == current) {
        next = 0u;
      }
      if (current == task_id) {
        if (previous != 0u) {
          const std::size_t previous_index = TaskSlot(previous);
          if (previous_index < control_->next_recv_waiter.size()) {
            control_->next_recv_waiter[previous_index] = next;
          }
        } else {
          control_->recv_waiter_head = next;
        }
        if (control_->recv_waiter_tail == task_id) {
          control_->recv_waiter_tail = previous;
        }
        control_->next_recv_waiter[current_index] = 0u;
        return;
      }
      previous = current;
      current = next;
    }
  }

  [[nodiscard]] bool PopSendWaiter(PendingSend* const pending) {
    while (control_->send_waiter_head != 0u) {
      const std::uint64_t front = control_->send_waiter_head;
      const std::size_t index = TaskSlot(front);
      if (index >= control_->next_send_waiter.size()) return false;
      control_->send_waiter_head = control_->next_send_waiter[index];
      if (control_->send_waiter_head == 0u) {
        control_->send_waiter_tail = 0u;
      }
      control_->next_send_waiter[index] = 0u;
      if (!SendWaiterActive(front)) {
        continue;
      }
      std::optional<T> value = std::move(control_->send_value_by_task[index]);
      RemoveSendWaiter(front);
      if (pending != nullptr) {
        *pending = PendingSend{front, std::move(value)};
      }
      return true;
    }
    return false;
  }

  [[nodiscard]] bool PopRecvWaiter(std::uint64_t* const task_id) noexcept {
    while (control_->recv_waiter_head != 0u) {
      const std::uint64_t front = control_->recv_waiter_head;
      const std::size_t index = TaskSlot(front);
      if (index >= control_->next_recv_waiter.size()) return false;
      control_->recv_waiter_head = control_->next_recv_waiter[index];
      if (control_->recv_waiter_head == 0u) {
        control_->recv_waiter_tail = 0u;
      }
      control_->next_recv_waiter[index] = 0u;
      if (!RecvWaiterActive(front)) {
        continue;
      }
      RemoveRecvWaiter(front);
      if (task_id != nullptr) {
        *task_id = front;
      }
      return true;
    }
    return false;
  }
