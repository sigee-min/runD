channel() noexcept = default;

channel(const channel &) = delete;
channel &operator=(const channel &) = delete;

channel(channel &&other) noexcept
    : code_(other.code_), capacity_(other.capacity_),
      control_(std::move(other.control_)), owns_(other.owns_) {
  other.code_ = ReasonCode::ChannelInvalid;
  other.capacity_ = 0u;
  other.owns_ = false;
}

channel &operator=(channel &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  Abandon();
  code_ = other.code_;
  capacity_ = other.capacity_;
  control_ = std::move(other.control_);
  owns_ = other.owns_;
  other.code_ = ReasonCode::ChannelInvalid;
  other.capacity_ = 0u;
  other.owns_ = false;
  return *this;
}

~channel() noexcept { Abandon(); }

[[nodiscard]] static channel make(std::size_t capacity) noexcept {
  channel out{};
  std::uint64_t channel_id = 0u;
  const ::rund::detail::task::ChannelDecision record =
      ::rund::detail::task::ChannelAccess::MakeChannelRecord(capacity,
                                                             &channel_id);
  out.capacity_ = capacity;
  out.code_ = record.status.code();
  if (!record.status) {
    (void)::rund::detail::task::ChannelAccess::FinishCurrentOperation();
    return out;
  }
  const ::rund::detail::task::ActiveState active_state =
      ::rund::detail::task::ChannelAccess::ActiveSchedulerState();
  const ::rund::detail::task::ChannelDecision complete =
      ::rund::detail::task::ChannelAccess::FinishCurrentOperation();
  if (!complete.status) {
    out.code_ = complete.status.code();
    return out;
  }
  try {
    auto control = std::make_shared<Control>();
    control->owner_scheduler_id = active_state.scheduler_id;
    control->channel_id = channel_id;
    control->capacity = capacity;
    control->task_capacity = active_state.task_capacity;
    control->buffer.resize(capacity);
    out.control_ = std::move(control);
    out.owns_ = true;
    out.code_ = ReasonCode::Ok;
  } catch (...) {
    (void)::rund::detail::task::ChannelAccess::ReleaseChannelRecord(channel_id,
                                                                    capacity);
    (void)::rund::detail::task::ChannelAccess::FinishCurrentOperation();
    out.control_.reset();
    out.code_ = ReasonCode::ChannelCapacityExceeded;
  }
  return out;
}

[[nodiscard]] explicit operator bool() const noexcept {
  return control_ != nullptr && code_ == ReasonCode::Ok;
}
[[nodiscard]] bool ok() const noexcept { return static_cast<bool>(*this); }
[[nodiscard]] ReasonCode code() const noexcept { return code_; }
[[nodiscard]] std::string_view error() const noexcept {
  return code_ == ReasonCode::Ok ? std::string_view{}
                                 : std::string_view{ReasonString(code_)};
}
[[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
