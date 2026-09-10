[[nodiscard]] std::uint32_t
filter_reduce(const std::span<const std::uint32_t> values) noexcept {
  std::uint32_t total = 0u;
  for (const std::uint32_t value : values) {
    if ((value & 7u) != 0u) {
      total += value;
    }
  }
  return total;
}

[[nodiscard]] std::uint32_t
window_sum_oracle(const std::span<const std::uint32_t> input,
                  const std::size_t radius) {
  if (input.empty()) {
    return 0u;
  }
  std::vector<std::uint64_t> prefix(input.size() + 1u, 0u);
  for (std::size_t index = 0u; index < input.size(); ++index) {
    prefix[index + 1u] = prefix[index] + input[index];
  }
  std::vector<std::uint32_t> output(input.size());
  for (std::size_t index = 0u; index < input.size(); ++index) {
    const std::size_t left = index > radius ? index - radius : 0u;
    const std::size_t right = std::min(input.size() - 1u, index + radius);
    const std::uint64_t left_repeat = radius > index ? radius - index : 0u;
    const std::uint64_t right_repeat = index + radius >= input.size()
                                           ? index + radius - input.size() + 1u
                                           : 0u;
    const std::uint64_t sum = prefix[right + 1u] - prefix[left] +
                              left_repeat * input.front() +
                              right_repeat * input.back();
    output[index] = static_cast<std::uint32_t>(sum);
  }
  return filter_reduce(output);
}

[[nodiscard]] std::uint32_t
pool_sum_oracle(const std::span<const std::uint32_t> input,
                const std::size_t width, const std::size_t stride) {
  if (input.empty()) {
    return 0u;
  }
  std::vector<std::uint64_t> prefix(input.size() + 1u, 0u);
  for (std::size_t index = 0u; index < input.size(); ++index) {
    prefix[index + 1u] = prefix[index] + input[index];
  }
  const std::size_t count = 1u + (input.size() - 1u) / stride;
  std::uint32_t total = 0u;
  for (std::size_t output = 0u; output < count; ++output) {
    const std::size_t begin = output * stride;
    const std::size_t end = std::min(input.size(), begin + width);
    const std::uint32_t value =
        static_cast<std::uint32_t>(prefix[end] - prefix[begin]);
    if ((value & 7u) != 0u) {
      total += value;
    }
  }
  return total;
}

[[nodiscard]] std::uint32_t
rolling_min_oracle(const std::span<const std::uint32_t> input,
                   const std::size_t radius) {
  if (input.empty()) {
    return 0u;
  }
  std::deque<std::size_t> deque;
  std::size_t inserted = 0u;
  std::uint32_t total = 0u;
  for (std::size_t output = 0u; output < input.size(); ++output) {
    const std::size_t right = std::min(input.size() - 1u, output + radius);
    while (inserted <= right) {
      while (!deque.empty() && input[deque.back()] >= input[inserted]) {
        deque.pop_back();
      }
      deque.push_back(inserted++);
    }
    const std::size_t left = output > radius ? output - radius : 0u;
    while (!deque.empty() && deque.front() < left) {
      deque.pop_front();
    }
    const std::uint32_t value = input[deque.front()];
    if ((value & 7u) != 0u) {
      total += value;
    }
  }
  return total;
}

[[nodiscard]] const char *range_name(const rund::compute::RangeKind kind) {
  using rund::compute::RangeKind;
  switch (kind) {
  case RangeKind::Direct:
    return "direct";
  case RangeKind::Shared:
    return "shared";
  case RangeKind::Prefix:
    return "prefix";
  case RangeKind::Tiled:
    return "tiled";
  case RangeKind::Block:
    return "block";
  }
  return "invalid";
}

template <class Program>
[[nodiscard]] bool inspect_range(const Program &program,
                                 rund::compute::RangeInfo &range) {
  const auto snapshot = program.ranges(std::span{&range, 1u});
  return snapshot.written == 1u && snapshot.total == 1u &&
         !snapshot.truncated() && range.stages != 0u &&
         (range.source.hi != 0u || range.source.lo != 0u) &&
         (range.execution.hi != 0u || range.execution.lo != 0u);
}
