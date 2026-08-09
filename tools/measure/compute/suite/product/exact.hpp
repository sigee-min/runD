template <class Author>
[[nodiscard]] bool
measure_exact(rund::compute::Device &device, const Backend backend,
              const ProductShape shape, const std::vector<std::uint32_t> &input,
              const std::uint32_t expected, Author &&author) {
  using namespace rund::compute;
  const auto cold_begin = Clock::now();
  auto flow = author();
  const auto authored = Clock::now();
  auto program = std::move(flow).compile();
  const auto compiled = Clock::now();
  if (!program) {
    return report_failure(backend, shape, "compile", program);
  }
  auto input_buffer = device.upload<std::uint32_t>(std::span{input});
  auto output_buffer = device.buffer<std::uint32_t>(1u);
  const auto uploaded = Clock::now();
  if (!input_buffer || !output_buffer) {
    return !input_buffer
               ? report_failure(backend, shape, "upload", input_buffer)
               : report_failure(backend, shape, "output", output_buffer);
  }
  auto pipeline_result =
      pipeline(device)
          .then(*program, read(*input_buffer), write(*output_buffer))
          .prepare();
  const auto prepared = Clock::now();
  if (!pipeline_result) {
    return report_failure(backend, shape, "prepare", pipeline_result);
  }
  const auto status = pipeline_result->run();
  const auto ran = Clock::now();
  if (!status) {
    return report_failure(backend, shape, "run", status);
  }
  TerminalEvidence cold_terminal{};
  const auto cold_read =
      read_terminal(*pipeline_result, *output_buffer, expected, cold_terminal);
  auto cold_profile = capture_terminal(*pipeline_result, cold_terminal);
  if (!cold_read || !cold_profile) {
    std::fprintf(stderr,
                 "product %s/%.*s/%.*s cold evidence mismatch: %u != %u; "
                 "submissions=%llu readbacks=%llu graph=%llu output=%llu\n",
                 Name(backend), static_cast<int>(shape.chain.size()),
                 shape.chain.data(), static_cast<int>(shape.name.size()),
                 shape.name.data(), cold_terminal.value, expected,
                 static_cast<unsigned long long>(cold_terminal.submissions),
                 static_cast<unsigned long long>(cold_terminal.readbacks),
                 static_cast<unsigned long long>(cold_terminal.graph_hash),
                 static_cast<unsigned long long>(cold_terminal.output_hash));
    return false;
  }
  const auto read_end = *cold_read;
  RangeInfo range{};
  if (!inspect_range(*program, range)) {
    std::fprintf(stderr, "product %s/%.*s/%.*s range evidence missing\n",
                 Name(backend), static_cast<int>(shape.chain.size()),
                 shape.chain.data(), static_cast<int>(shape.name.size()),
                 shape.name.data());
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
  print_cold(backend, shape, range, times, *cold_profile, cold_terminal, plan,
             input.size() * sizeof(std::uint32_t), sizeof(std::uint32_t));

  if (!condition(*pipeline_result)) {
    return false;
  }
  std::vector<double> samples;
  samples.reserve(WarmSamples);
  const Status began = pipeline_result->begin_samples();
  if (!began) {
    return report_failure(backend, shape, "warm-begin", began);
  }
  for (std::size_t sample = 0u; sample < WarmSamples; ++sample) {
    const auto begin = Clock::now();
    const Status ran_status = pipeline_result->run();
    const auto end = Clock::now();
    if (!ran_status) {
      return report_failure(backend, shape, "warm", ran_status);
    }
    samples.push_back(elapsed_us(begin, end));
  }
  const Status ended = pipeline_result->end_samples();
  if (!ended) {
    return report_failure(backend, shape, "warm-end", ended);
  }
  std::sort(samples.begin(), samples.end());
  TerminalEvidence terminal{};
  const auto terminal_status =
      read_terminal(*pipeline_result, *output_buffer, expected, terminal);
  auto terminal_profile = capture_terminal(*pipeline_result, terminal);
  if (!terminal_status || !terminal_profile ||
      !terminal_profile->execution().pipeline.samples_clean(WarmSamples)) {
    std::fprintf(stderr, "product %s/%.*s/%.*s warm evidence failed\n",
                 Name(backend), static_cast<int>(shape.chain.size()),
                 shape.chain.data(), static_cast<int>(shape.name.size()),
                 shape.name.data());
    return false;
  }
  print_warm(backend, shape, range, samples, *terminal_profile, terminal, plan);
  return true;
}
