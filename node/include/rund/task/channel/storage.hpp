  ReasonCode code_ = ReasonCode::ChannelInvalid;
  std::size_t capacity_ = 0u;
  std::shared_ptr<Control> control_{};
  bool owns_ = false;
