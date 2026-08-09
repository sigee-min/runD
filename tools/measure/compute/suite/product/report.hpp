template <class Result>
[[nodiscard]] bool
report_failure(const Backend backend, const ProductShape &shape,
               const std::string_view phase, const Result &result) {
  std::fprintf(stderr, "product %s/%.*s/%.*s %.*s failed: %.*s\n",
               Name(backend), static_cast<int>(shape.chain.size()),
               shape.chain.data(), static_cast<int>(shape.name.size()),
               shape.name.data(), static_cast<int>(phase.size()), phase.data(),
               static_cast<int>(result.error().size()), result.error().data());
  return false;
}

void print_cold(const Backend backend, const ProductShape &shape,
                const rund::compute::RangeInfo &range, const PhaseTimes &times,
                const rund::compute::telemetry::Profile &profile,
                const TerminalEvidence &terminal,
                const rund::compute::PipelinePlan &plan,
                const std::uint64_t logical_upload_bytes,
                const std::uint64_t logical_readback_bytes) {
  const rund::compute::Stats &run = profile.execution();
  const rund::compute::MemoryStats &memory = profile.memory();
  const auto field = [](const std::uint64_t value) {
    std::printf(",%llu", static_cast<unsigned long long>(value));
  };
  std::printf("product_cold,%s,%.*s,%.*s,ok,%zu,%zu,%zu,%zu,%zu,%s,%u,%u,%u,"
              "%llu,%llu,%llu,%llu,%llu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
              Name(backend), static_cast<int>(shape.chain.size()),
              shape.chain.data(), static_cast<int>(shape.name.size()),
              shape.name.data(), shape.capacity, shape.active, shape.radius,
              shape.window, shape.stride, range_name(range.kind), range.width,
              range.stages, range.shared_capacity,
              static_cast<unsigned long long>(range.scratch_bytes),
              static_cast<unsigned long long>(range.source.hi),
              static_cast<unsigned long long>(range.source.lo),
              static_cast<unsigned long long>(range.execution.hi),
              static_cast<unsigned long long>(range.execution.lo),
              times.first_result_us, times.author_us, times.compile_us,
              times.upload_us, times.prepare_us, times.run_us, times.read_us);
  field(run.dispatches);
  field(run.command_submits);
  field(terminal.submissions);
  field(terminal.readbacks);
  field(logical_upload_bytes);
  field(logical_readback_bytes);
  field(terminal.downloaded_bytes);
  field(plan.peak_bytes);
  field(memory.resident.peak);
  field(plan.scratch_payload_bytes);
  field(plan.scratch_bytes);
  field(run.pipeline_compiles);
  field(run.pipeline_cache_hits);
  field(plan.prepared_template_count);
  field(terminal.graph_hash);
  field(terminal.output_hash);
  std::printf(",%u\n", terminal.value);
}

void print_warm(const Backend backend, const ProductShape &shape,
                const rund::compute::RangeInfo &range,
                const std::vector<double> &sorted,
                const rund::compute::telemetry::Profile &profile,
                const TerminalEvidence &terminal,
                const rund::compute::PipelinePlan &plan) {
  const rund::compute::Stats &warm = profile.execution();
  const rund::compute::MemoryStats &memory = profile.memory();
  const double p50 = percentile(sorted, 50u, 100u);
  const double p95 = percentile(sorted, 95u, 100u);
  const double rate =
      p50 == 0.0 ? 0.0 : static_cast<double>(shape.active) * 1.0e6 / p50;
  const auto field = [](const std::uint64_t value) {
    std::printf(",%llu", static_cast<unsigned long long>(value));
  };
  std::printf("product_warm,%s,%.*s,%.*s,ok,%zu,%zu,%zu,%zu,%zu,%s,%u,%u,%u,"
              "%llu,%llu,%llu,%llu,%llu,%zu,%.3f,%.3f,%.3f",
              Name(backend), static_cast<int>(shape.chain.size()),
              shape.chain.data(), static_cast<int>(shape.name.size()),
              shape.name.data(), shape.capacity, shape.active, shape.radius,
              shape.window, shape.stride, range_name(range.kind), range.width,
              range.stages, range.shared_capacity,
              static_cast<unsigned long long>(range.scratch_bytes),
              static_cast<unsigned long long>(range.source.hi),
              static_cast<unsigned long long>(range.source.lo),
              static_cast<unsigned long long>(range.execution.hi),
              static_cast<unsigned long long>(range.execution.lo),
              sorted.size(), p50, p95, rate);
  field(warm.dispatches);
  field(warm.command_submits);
  field(terminal.submissions);
  field(terminal.readbacks);
  field(terminal.downloaded_bytes);
  field(plan.peak_bytes);
  field(memory.resident.peak);
  field(plan.scratch_payload_bytes);
  field(plan.scratch_bytes);
  field(warm.pipeline.sampled_runs);
  field(warm.pipeline.clean_runs);
  field(warm.pipeline_cache_hits);
  field(terminal.graph_hash);
  field(terminal.output_hash);
  std::printf(",%u\n", terminal.value);
}

[[nodiscard]] rund::compute::Result<Clock::time_point>
read_terminal(rund::compute::Pipeline &pipeline,
              const rund::compute::Buffer<std::uint32_t> &out,
              const std::uint32_t expected, TerminalEvidence &terminal) {
  std::array<std::uint32_t, 1u> value{};
  const auto read = pipeline.read(out, std::span{value});
  const auto completed = Clock::now();
  if (!read) {
    return rund::compute::Result<Clock::time_point>::fail(read.reason());
  }
  terminal.value = value[0u];
  return value[0u] == expected
             ? rund::compute::Result<Clock::time_point>::success(completed)
             : rund::compute::Result<Clock::time_point>::fail(
                   rund::compute::Reason::CompletionInvalid);
}

[[nodiscard]] rund::compute::Result<rund::compute::telemetry::Profile>
capture_terminal(rund::compute::Pipeline &pipeline,
                 TerminalEvidence &terminal) {
  auto observed = pipeline.profile();
  if (!observed) {
    return observed;
  }
  const rund::compute::Stats &after = observed->execution();
  terminal = TerminalEvidence{
      .submissions = after.transfer_submissions.device_to_host,
      .readbacks = after.download_events,
      .downloaded_bytes = after.downloaded_bytes,
      .graph_hash = after.graph_hash,
      .output_hash = after.output_hash,
      .value = terminal.value,
  };
  if (terminal.readbacks != 1u || terminal.graph_hash == 0u ||
      terminal.output_hash == 0u) {
    return rund::compute::Result<rund::compute::telemetry::Profile>::fail(
        rund::compute::Reason::CompletionInvalid);
  }
  return observed;
}
