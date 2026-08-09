constexpr std::size_t WarmSamples = 60u;
constexpr std::size_t WarmConditionRuns = WarmSamples;
constexpr std::size_t BoundedCapacity = 1u << 18u;
constexpr std::size_t BoundedRadius = 1024u;

struct ProductShape final {
  std::string_view chain{};
  std::string_view name{};
  std::size_t capacity{};
  std::size_t active{};
  std::size_t radius{};
  std::size_t window{};
  std::size_t stride{};
};

struct PhaseTimes final {
  double first_result_us{};
  double author_us{};
  double compile_us{};
  double upload_us{};
  double prepare_us{};
  double run_us{};
  double read_us{};
};

struct TerminalEvidence final {
  std::uint64_t submissions{};
  std::uint64_t readbacks{};
  std::uint64_t downloaded_bytes{};
  std::uint64_t graph_hash{};
  std::uint64_t output_hash{};
  std::uint32_t value{};
};

[[nodiscard]] double elapsed_us(const Clock::time_point begin,
                                const Clock::time_point end) noexcept {
  return std::chrono::duration<double, std::micro>(end - begin).count();
}

[[nodiscard]] double percentile(const std::vector<double> &sorted,
                                const std::size_t numerator,
                                const std::size_t denominator) noexcept {
  if (sorted.empty() || denominator == 0u) {
    return 0.0;
  }
  const std::size_t rank =
      (sorted.size() * numerator + denominator - 1u) / denominator;
  return sorted[std::max<std::size_t>(1u, rank) - 1u];
}

[[nodiscard]] bool condition(rund::compute::Pipeline &pipeline) {
  for (std::size_t run = 0u; run < WarmConditionRuns; ++run) {
    if (!pipeline.run()) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr std::uint32_t map_value(const std::uint32_t value) {
  return (value & 3u) + 1u;
}

[[nodiscard]] std::uint32_t input_value(const std::size_t index) noexcept {
  const std::uint32_t narrow = static_cast<std::uint32_t>(index);
  return (narrow * 2654435761u) ^ (narrow >> 7u) ^ 0x9e3779b9u;
}

void fill_input(std::vector<std::uint32_t> &values,
                const std::size_t active) noexcept {
  for (std::size_t index = 0u; index < values.size(); ++index) {
    values[index] =
        index < active
            ? input_value(index)
            : (index % 2u == 0u ? 0u
                                : std::numeric_limits<std::uint32_t>::max());
  }
}

[[nodiscard]] std::vector<std::uint32_t>
mapped_prefix(const std::span<const std::uint32_t> input,
              const std::size_t active) {
  std::vector<std::uint32_t> mapped(active);
  for (std::size_t index = 0u; index < active; ++index) {
    mapped[index] = map_value(input[index]);
  }
  return mapped;
}
