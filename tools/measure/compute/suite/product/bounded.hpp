[[nodiscard]] bool measure_bounded(rund::compute::Device &device,
                                   rund::compute::ProgramCache &cache,
                                   const Backend backend) {
  using namespace rund::compute;
  const ProductShape cold_shape{
      .chain = "rolling",
      .name = "m262144_r1024_n0",
      .capacity = BoundedCapacity,
      .active = 0u,
      .radius = BoundedRadius,
      .window = BoundedRadius * 2u + 1u,
      .stride = 1u,
  };
  std::vector<std::uint32_t> input(BoundedCapacity);
  fill_input(input, 0u);
  std::array<std::uint32_t, 1u> count{0u};

  const auto cold_begin = Clock::now();
  auto flow = on(device, cache)
                  .input<Bounded<std::uint32_t>>(BoundedCapacity)
                  .map("product-rolling-map",
                       [](auto value) { return (value & 3u) + 1u; })
                  .window({.op = Window::Min, .radius = BoundedRadius})
                  .filter([](auto value) { return (value & 7u) != 0u; })
                  .reduce(Reduce::Sum);
  const auto authored = Clock::now();
  auto program = std::move(flow).compile();
  const auto compiled = Clock::now();
  if (!program) {
    return report_failure(backend, cold_shape, "compile", program);
  }
  auto input_buffer = device.upload<std::uint32_t>(std::span{input});
  auto count_buffer = device.upload<std::uint32_t>(std::span{count});
  auto output_buffer = device.buffer<std::uint32_t>(1u);
  const auto uploaded = Clock::now();
  if (!input_buffer || !count_buffer || !output_buffer) {
    std::fprintf(stderr, "product %s/rolling bounded buffer setup failed\n",
                 Name(backend));
    return false;
  }
  auto pipeline_result = pipeline(device)
                             .then(*program, read(*input_buffer, *count_buffer),
                                   write(*output_buffer))
                             .prepare();
  const auto prepared = Clock::now();
  if (!pipeline_result) {
    return report_failure(backend, cold_shape, "prepare", pipeline_result);
  }
  const Status status = pipeline_result->run();
  const auto ran = Clock::now();
  if (!status) {
    return report_failure(backend, cold_shape, "run", status);
  }
  TerminalEvidence cold_terminal{};
  const auto cold_read =
      read_terminal(*pipeline_result, *output_buffer, 0u, cold_terminal);
  auto cold_profile = capture_terminal(*pipeline_result, cold_terminal);
  if (!cold_read || !cold_profile) {
    return false;
  }
  const auto read_end = *cold_read;
  RangeInfo range{};
  if (!inspect_range(*program, range)) {
    return false;
  }
  const PipelinePlan plan = pipeline_result->plan();
  const PhaseTimes times{
      .first_result_us = elapsed_us(cold_begin, read_end),
      .author_us = elapsed_us(cold_begin, authored),
      .compile_us = elapsed_us(authored, compiled),
      .upload_us = elapsed_us(compiled, uploaded),
      .prepare_us = elapsed_us(uploaded, prepared),
      .run_us = elapsed_us(prepared, ran),
      .read_us = elapsed_us(ran, read_end),
  };
  print_cold(backend, cold_shape, range, times, *cold_profile, cold_terminal,
             plan, input.size() * sizeof(std::uint32_t) + sizeof(std::uint32_t),
             sizeof(std::uint32_t));

  constexpr std::array<std::size_t, 4u> active_counts{
      0u, 257u, BoundedCapacity / 2u, BoundedCapacity};
  for (std::size_t phase = 0u; phase < active_counts.size(); ++phase) {
    const std::size_t active = active_counts[phase];
    if (!condition(*pipeline_result)) {
      std::fprintf(stderr, "product %s/rolling warm conditioning failed\n",
                   Name(backend));
      return false;
    }
    std::vector<double> samples;
    samples.reserve(WarmSamples);
    const Status began = pipeline_result->begin_samples();
    if (!began) {
      return report_failure(backend, cold_shape, "warm-begin", began);
    }
    for (std::size_t sample = 0u; sample < WarmSamples; ++sample) {
      const auto begin = Clock::now();
      const Status ran_status = pipeline_result->run();
      const auto end = Clock::now();
      if (!ran_status) {
        return report_failure(backend, cold_shape, "warm", ran_status);
      }
      samples.push_back(elapsed_us(begin, end));
    }
    const Status ended = pipeline_result->end_samples();
    if (!ended) {
      return report_failure(backend, cold_shape, "warm-end", ended);
    }
    std::sort(samples.begin(), samples.end());
    const auto mapped = mapped_prefix(input, active);
    const std::uint32_t expected =
        rolling_min_oracle(std::span{mapped}, BoundedRadius);
    TerminalEvidence terminal{};
    const auto terminal_status =
        read_terminal(*pipeline_result, *output_buffer, expected, terminal);
    auto terminal_profile = capture_terminal(*pipeline_result, terminal);
    if (!terminal_status || !terminal_profile ||
        !terminal_profile->execution().pipeline.samples_clean(WarmSamples)) {
      std::fprintf(stderr,
                   "product %s/rolling terminal evidence n=%zu failed\n",
                   Name(backend), active);
      return false;
    }
    if (phase + 1u != active_counts.size()) {
      const std::size_t next_active = active_counts[phase + 1u];
      std::vector<std::uint32_t> next(BoundedCapacity);
      fill_input(next, next_active);
      const std::array<std::uint32_t, 1u> next_count{
          static_cast<std::uint32_t>(next_active)};
      bool observed = false;
      const Status transitioned = host_feedback(
          *pipeline_result, 2u,
          [&](HostIteration &iteration) noexcept -> Status {
            if (iteration.completed() != 1u) {
              return Status::success();
            }
            Status changed = iteration.write(*input_buffer, std::span{next});
            if (!changed) {
              return changed;
            }
            changed = iteration.write(*count_buffer, std::span{next_count});
            if (!changed) {
              return changed;
            }
            observed = true;
            return changed;
          });
      if (!transitioned || !observed) {
        std::fprintf(stderr,
                     "product %s/rolling transition n=%zu failed: status=%.*s "
                     "observed=%u value=%u/%u\n",
                     Name(backend), active,
                     static_cast<int>(transitioned.error().size()),
                     transitioned.error().data(), observed ? 1u : 0u,
                     terminal.value, expected);
        return false;
      }
      input = std::move(next);
    }
    const std::string_view name = phase == 0u   ? "m262144_r1024_n0"
                                  : phase == 1u ? "m262144_r1024_n257"
                                  : phase == 2u ? "m262144_r1024_n131072"
                                                : "m262144_r1024_n262144";
    const ProductShape shape{
        .chain = "rolling",
        .name = name,
        .capacity = BoundedCapacity,
        .active = active,
        .radius = BoundedRadius,
        .window = BoundedRadius * 2u + 1u,
        .stride = 1u,
    };
    print_warm(backend, shape, range, samples, *terminal_profile, terminal,
               plan);
  }
  return true;
}
