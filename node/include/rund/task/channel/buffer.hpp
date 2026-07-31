  [[nodiscard]] std::size_t BufferSize() const noexcept {
    return control_->buffer_size;
  }

  [[nodiscard]] bool BufferEmpty() const noexcept {
    return control_->buffer_size == 0u;
  }

  [[nodiscard]] bool BufferFull() const noexcept {
    return control_->buffer_size == control_->capacity;
  }

  void PushBuffered(T&& value) {
    const std::size_t index =
        (control_->buffer_head + control_->buffer_size) % control_->capacity;
    control_->buffer[index].emplace(std::move(value));
    ++control_->buffer_size;
  }

  [[nodiscard]] T PopBuffered() {
    std::optional<T>& slot = control_->buffer[control_->buffer_head];
    T value = std::move(*slot);
    slot.reset();
    --control_->buffer_size;
    control_->buffer_head = (control_->buffer_head + 1u) % control_->capacity;
    return value;
  }

  void ClearBuffered() noexcept {
    for (std::optional<T>& slot : control_->buffer) {
      slot.reset();
    }
    control_->buffer_head = 0u;
    control_->buffer_size = 0u;
  }
