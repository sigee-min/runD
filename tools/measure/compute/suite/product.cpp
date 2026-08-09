#include "core.hpp"

#include <rund/compute/cache.hpp>
#include <rund/compute/pipeline.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <limits>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace rund::measure::compute {
namespace {

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

struct WarmEvidence final {
  WarmCounters counters{};
  std::uint64_t cache_hits{};
  std::uint64_t dispatches{};
  std::uint64_t submissions{};
  std::uint64_t rebindings{};
  std::uint64_t graph_hash{};
  bool initialized{};
  bool consistent{true};

  void observe(const rund::compute::Stats &stats) noexcept {
    counters.observe(stats);
    ::rund::detail::counter::Accumulate(cache_hits, stats.pipeline_cache_hits);
    ::rund::detail::counter::Accumulate(rebindings,
                                        stats.pipeline.rebinding_count);
    if (!initialized) {
      initialized = true;
      dispatches = stats.dispatches;
      submissions = stats.command_submits;
      graph_hash = stats.graph_hash;
    } else if (dispatches != stats.dispatches ||
               submissions != stats.command_submits ||
               graph_hash != stats.graph_hash) {
      consistent = false;
    }
  }

  [[nodiscard]] bool clean(const Backend backend) const noexcept {
    const std::uint64_t expected_submits = backend == Backend::Cpu ? 0u : 1u;
    return initialized && consistent && counters.zero() && rebindings == 0u &&
           submissions == expected_submits && graph_hash != 0u;
  }
};

struct TerminalEvidence final {
  std::uint64_t submissions{};
  std::uint64_t readbacks{};
  std::uint64_t downloaded_bytes{};
  std::uint64_t transition_upload_bytes{};
  std::uint64_t graph_hash{};
  std::uint64_t output_hash{};
  std::uint32_t value{};
};

[[nodiscard]] double elapsed_us(const Clock::time_point begin,
                                const Clock::time_point end) noexcept {
  return std::chrono::duration<double, std::micro>(end - begin).count();
}

[[nodiscard]] constexpr std::uint64_t delta(const std::uint64_t after,
                                            const std::uint64_t before) {
  return after >= before ? after - before : 0u;
}

[[nodiscard]] constexpr std::uint64_t
terminal_submissions(const Backend backend) noexcept {
  return backend == Backend::Vulkan ? 1u : 0u;
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
                const rund::compute::ProgramCache::Stats cache_before,
                const rund::compute::ProgramCache::Stats cache_after,
                const rund::compute::Stats &run,
                const TerminalEvidence &terminal,
                const rund::compute::PipelinePlan &plan,
                const rund::compute::MemoryStats &memory,
                const std::uint64_t upload_bytes,
                const std::uint64_t logical_upload_bytes,
                const std::uint64_t logical_readback_bytes) {
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
  field(upload_bytes);
  field(terminal.downloaded_bytes);
  field(plan.peak_bytes);
  field(memory.resident.peak);
  field(plan.scratch_payload_bytes);
  field(plan.scratch_bytes);
  field(run.pipeline_compiles);
  field(run.pipeline_cache_hits);
  field(delta(cache_after.hits, cache_before.hits));
  field(delta(cache_after.misses, cache_before.misses));
  field(plan.prepared_template_count);
  field(terminal.graph_hash);
  field(terminal.output_hash);
  std::printf(",%u\n", terminal.value);
}

void print_warm(const Backend backend, const ProductShape &shape,
                const rund::compute::RangeInfo &range,
                const std::vector<double> &sorted, const WarmEvidence &warm,
                const TerminalEvidence &terminal,
                const rund::compute::PipelinePlan &plan,
                const rund::compute::MemoryStats &memory) {
  const double p50 = percentile(sorted, 50u, 100u);
  const double p95 = percentile(sorted, 95u, 100u);
  const double rate =
      p50 == 0.0 ? 0.0 : static_cast<double>(shape.active) * 1.0e6 / p50;
  const std::uint64_t allocations = warm.counters.buffer_allocations +
                                    warm.counters.descriptor_pool_creations +
                                    warm.counters.descriptor_set_allocations;
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
  field(warm.submissions);
  field(terminal.submissions);
  field(terminal.readbacks);
  field(terminal.transition_upload_bytes);
  field(terminal.downloaded_bytes);
  field(plan.peak_bytes);
  field(memory.resident.peak);
  field(plan.scratch_payload_bytes);
  field(plan.scratch_bytes);
  field(warm.counters.pipeline_compiles);
  field(warm.cache_hits);
  field(warm.counters.buffer_allocations);
  field(warm.counters.descriptor_pool_creations);
  field(warm.counters.descriptor_set_allocations);
  field(allocations);
  field(warm.counters.uploaded_bytes);
  field(warm.counters.download_events);
  field(warm.counters.downloaded_bytes);
  field(warm.rebindings);
  field(terminal.graph_hash);
  field(terminal.output_hash);
  std::printf(",%u\n", terminal.value);
}

[[nodiscard]] bool
terminal_read(const Backend backend, rund::compute::Pipeline &pipeline,
              const rund::compute::Buffer<std::uint32_t> &out,
              const std::uint32_t expected, TerminalEvidence &terminal) {
  std::array<std::uint32_t, 1u> value{};
  const auto before = pipeline.stats();
  const auto read = pipeline.read(out, std::span{value});
  const auto after = pipeline.stats();
  terminal = TerminalEvidence{
      .submissions = delta(after.command_submits, before.command_submits),
      .readbacks = delta(after.download_events, before.download_events),
      .downloaded_bytes =
          delta(after.downloaded_bytes, before.downloaded_bytes),
      .graph_hash = after.graph_hash,
      .output_hash = after.output_hash,
      .value = value[0u],
  };
  const std::uint64_t expected_submissions = terminal_submissions(backend);
  return read && value[0u] == expected && terminal.readbacks == 1u &&
         terminal.submissions == expected_submissions &&
         terminal.graph_hash != 0u && terminal.output_hash != 0u;
}

template <class Author>
[[nodiscard]] bool
measure_exact(rund::compute::Device &device, rund::compute::ProgramCache &cache,
              const Backend backend, const ProductShape shape,
              const std::vector<std::uint32_t> &input,
              const std::uint32_t expected, Author &&author) {
  using namespace rund::compute;
  const auto cache_before = cache.stats();
  const auto cold_begin = Clock::now();
  auto flow = author();
  const auto authored = Clock::now();
  auto program = std::move(flow).compile();
  const auto compiled = Clock::now();
  if (!program) {
    return report_failure(backend, shape, "compile", program);
  }
  const auto memory_before = device.memory();
  auto input_buffer = device.upload<std::uint32_t>(std::span{input});
  auto output_buffer = device.buffer<std::uint32_t>(1u);
  const auto uploaded = Clock::now();
  const auto memory_after = device.memory();
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
  const auto run_stats = pipeline_result->stats();
  TerminalEvidence cold_terminal{};
  if (!terminal_read(backend, *pipeline_result, *output_buffer, expected,
                     cold_terminal)) {
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
  const auto read_end = Clock::now();
  RangeInfo range{};
  if (!inspect_range(*program, range)) {
    std::fprintf(stderr, "product %s/%.*s/%.*s range evidence missing\n",
                 Name(backend), static_cast<int>(shape.chain.size()),
                 shape.chain.data(), static_cast<int>(shape.name.size()),
                 shape.name.data());
    return false;
  }
  const PipelinePlan plan = pipeline_result->plan();
  const MemoryStats memory = pipeline_result->memory();
  const auto cache_after = cache.stats();
  const PhaseTimes times{
      .first_result_us = elapsed_us(cold_begin, read_end),
      .author_us = elapsed_us(cold_begin, authored),
      .compile_us = elapsed_us(authored, compiled),
      .upload_us = elapsed_us(compiled, uploaded),
      .prepare_us = elapsed_us(uploaded, prepared),
      .run_us = elapsed_us(prepared, ran),
      .read_us = elapsed_us(ran, read_end),
  };
  print_cold(backend, shape, range, times, cache_before, cache_after, run_stats,
             cold_terminal, plan, memory,
             delta(memory_after.transfer.cumulative,
                   memory_before.transfer.cumulative),
             input.size() * sizeof(std::uint32_t), sizeof(std::uint32_t));

  if (!condition(*pipeline_result)) {
    return false;
  }
  std::vector<double> samples;
  samples.reserve(WarmSamples);
  WarmEvidence warm{};
  for (std::size_t sample = 0u; sample < WarmSamples; ++sample) {
    const auto begin = Clock::now();
    const Status ran_status = pipeline_result->run();
    const auto end = Clock::now();
    if (!ran_status) {
      return report_failure(backend, shape, "warm", ran_status);
    }
    const Stats stats = pipeline_result->stats();
    warm.observe(stats);
    samples.push_back(elapsed_us(begin, end));
  }
  std::sort(samples.begin(), samples.end());
  TerminalEvidence terminal{};
  if (!terminal_read(backend, *pipeline_result, *output_buffer, expected,
                     terminal) ||
      !warm.clean(backend)) {
    std::fprintf(stderr, "product %s/%.*s/%.*s warm evidence failed\n",
                 Name(backend), static_cast<int>(shape.chain.size()),
                 shape.chain.data(), static_cast<int>(shape.name.size()),
                 shape.name.data());
    return false;
  }
  print_warm(backend, shape, range, samples, warm, terminal, plan, memory);
  return true;
}

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

  const auto cache_before = cache.stats();
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
  const auto memory_before = device.memory();
  auto input_buffer = device.upload<std::uint32_t>(std::span{input});
  auto count_buffer = device.upload<std::uint32_t>(std::span{count});
  auto output_buffer = device.buffer<std::uint32_t>(1u);
  const auto uploaded = Clock::now();
  const auto memory_after = device.memory();
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
  const Stats run_stats = pipeline_result->stats();
  TerminalEvidence cold_terminal{};
  if (!terminal_read(backend, *pipeline_result, *output_buffer, 0u,
                     cold_terminal)) {
    return false;
  }
  const auto read_end = Clock::now();
  RangeInfo range{};
  if (!inspect_range(*program, range)) {
    return false;
  }
  const PipelinePlan plan = pipeline_result->plan();
  const MemoryStats memory = pipeline_result->memory();
  const auto cache_after = cache.stats();
  const PhaseTimes times{
      .first_result_us = elapsed_us(cold_begin, read_end),
      .author_us = elapsed_us(cold_begin, authored),
      .compile_us = elapsed_us(authored, compiled),
      .upload_us = elapsed_us(compiled, uploaded),
      .prepare_us = elapsed_us(uploaded, prepared),
      .run_us = elapsed_us(prepared, ran),
      .read_us = elapsed_us(ran, read_end),
  };
  print_cold(backend, cold_shape, range, times, cache_before, cache_after,
             run_stats, cold_terminal, plan, memory,
             delta(memory_after.transfer.cumulative,
                   memory_before.transfer.cumulative),
             input.size() * sizeof(std::uint32_t) + sizeof(std::uint32_t),
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
    WarmEvidence warm{};
    for (std::size_t sample = 0u; sample < WarmSamples; ++sample) {
      const auto begin = Clock::now();
      const Status ran_status = pipeline_result->run();
      const auto end = Clock::now();
      if (!ran_status) {
        return report_failure(backend, cold_shape, "warm", ran_status);
      }
      warm.observe(pipeline_result->stats());
      samples.push_back(elapsed_us(begin, end));
    }
    std::sort(samples.begin(), samples.end());
    const auto mapped = mapped_prefix(input, active);
    const std::uint32_t expected =
        rolling_min_oracle(std::span{mapped}, BoundedRadius);
    TerminalEvidence terminal{};
    if (phase + 1u == active_counts.size()) {
      if (!terminal_read(backend, *pipeline_result, *output_buffer, expected,
                         terminal)) {
        std::fprintf(stderr,
                     "product %s/rolling terminal evidence n=%zu failed\n",
                     Name(backend), active);
        return false;
      }
    } else {
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
            const auto before = iteration.stats();
            std::array<std::uint32_t, 1u> value{};
            Status changed = iteration.read(*output_buffer, std::span{value});
            if (!changed || value[0u] != expected) {
              return Status::fail(Reason::CompletionInvalid);
            }
            const auto after = iteration.stats();
            terminal.submissions =
                delta(after.command_submits, before.command_submits);
            terminal.readbacks =
                delta(after.download_events, before.download_events);
            terminal.downloaded_bytes =
                delta(after.downloaded_bytes, before.downloaded_bytes);
            terminal.graph_hash = after.graph_hash;
            terminal.output_hash = after.output_hash;
            terminal.value = value[0u];
            changed = iteration.write(*input_buffer, std::span{next});
            if (!changed) {
              return changed;
            }
            changed = iteration.write(*count_buffer, std::span{next_count});
            terminal.transition_upload_bytes = iteration.write_stats().bytes;
            observed = static_cast<bool>(changed);
            return changed;
          });
      const std::uint64_t expected_submissions = terminal_submissions(backend);
      if (!transitioned || !observed || terminal.readbacks != 1u ||
          terminal.submissions != expected_submissions) {
        std::fprintf(
            stderr,
            "product %s/rolling transition n=%zu failed: status=%.*s "
            "observed=%u submissions=%llu/%llu readbacks=%llu value=%u/%u\n",
            Name(backend), active,
            static_cast<int>(transitioned.error().size()),
            transitioned.error().data(), observed ? 1u : 0u,
            static_cast<unsigned long long>(terminal.submissions),
            static_cast<unsigned long long>(expected_submissions),
            static_cast<unsigned long long>(terminal.readbacks), terminal.value,
            expected);
        return false;
      }
      input = std::move(next);
    }
    if (!warm.clean(backend)) {
      std::fprintf(
          stderr,
          "product %s/rolling warm n=%zu dirty: consistent=%u "
          "dispatch=%llu submit=%llu graph=%llu alloc=%llu/%llu/%llu "
          "compile=%llu upload=%llu download=%llu rebind=%llu\n",
          Name(backend), active, warm.consistent ? 1u : 0u,
          static_cast<unsigned long long>(warm.dispatches),
          static_cast<unsigned long long>(warm.submissions),
          static_cast<unsigned long long>(warm.graph_hash),
          static_cast<unsigned long long>(warm.counters.buffer_allocations),
          static_cast<unsigned long long>(
              warm.counters.descriptor_pool_creations),
          static_cast<unsigned long long>(
              warm.counters.descriptor_set_allocations),
          static_cast<unsigned long long>(warm.counters.pipeline_compiles),
          static_cast<unsigned long long>(warm.counters.uploaded_bytes),
          static_cast<unsigned long long>(warm.counters.downloaded_bytes),
          static_cast<unsigned long long>(warm.rebindings));
      return false;
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
    print_warm(backend, shape, range, samples, warm, terminal, plan, memory);
  }
  return true;
}

} // namespace

void PrintProductColumns() {
  std::fputs(
      "product_cold_columns,backend,chain,shape,status,capacity,active_count,"
      "radius,window_size,stride,candidate,width,stages,shared_capacity,"
      "candidate_scratch_bytes,source_hi,source_lo,execution_hi,execution_lo,"
      "first_result_us,author_us,compile_us,upload_us,prepare_us,run_us,"
      "read_us,dispatches,run_submissions,terminal_submissions,"
      "terminal_readbacks,logical_upload_bytes,logical_readback_bytes,"
      "transfer_upload_bytes,downloaded_bytes,peak_retained_bytes,"
      "resident_peak_bytes,scratch_payload_bytes,scratch_backing_bytes,"
      "pipeline_compiles,pipeline_cache_hits,program_cache_hits,"
      "program_cache_misses,prepared_templates,graph_hash,output_hash,result\n",
      stdout);
  std::fputs(
      "product_warm_columns,backend,chain,shape,status,capacity,active_count,"
      "radius,window_size,stride,candidate,width,stages,shared_capacity,"
      "candidate_scratch_bytes,source_hi,source_lo,execution_hi,execution_lo,"
      "samples,warm_p50_us,warm_p95_us,active_elements_per_s,"
      "dispatches_per_run,submissions_per_run,terminal_submissions,"
      "terminal_readbacks,transition_upload_bytes,downloaded_bytes,"
      "peak_retained_bytes,resident_peak_bytes,scratch_payload_bytes,"
      "scratch_backing_bytes,warm_pipeline_compiles,warm_cache_hits,"
      "warm_buffer_allocations,warm_descriptor_pool_creations,"
      "warm_descriptor_set_allocations,warm_allocations,warm_uploaded_bytes,"
      "warm_download_events,warm_downloaded_bytes,warm_rebindings,graph_hash,"
      "output_hash,result\n",
      stdout);
}

bool ProductScenarios(const Backend backend) {
  using namespace rund::compute;
  auto opened = open(TargetFor(backend));
  if (!opened) {
    std::fprintf(stderr, "product %s device open failed: %.*s\n", Name(backend),
                 static_cast<int>(opened.error().size()),
                 opened.error().data());
    return false;
  }
  Device device = std::move(opened).value();
  auto cache_result = program_cache(device, 32u);
  if (!cache_result) {
    std::fprintf(stderr, "product %s cache failed: %.*s\n", Name(backend),
                 static_cast<int>(cache_result.error().size()),
                 cache_result.error().data());
    return false;
  }
  ProgramCache cache = std::move(cache_result).value();
  bool ok = true;
  const auto exact_window = [&](const std::size_t count,
                                const std::size_t radius,
                                const std::string_view name) {
    std::vector<std::uint32_t> input(count);
    fill_input(input, count);
    const auto mapped = mapped_prefix(input, count);
    const std::uint32_t expected = window_sum_oracle(mapped, radius);
    const ProductShape shape{.chain = "window",
                             .name = name,
                             .capacity = count,
                             .active = count,
                             .radius = radius,
                             .window = radius * 2u + 1u,
                             .stride = 1u};
    return measure_exact(device, cache, backend, shape, input, expected, [&] {
      return on(device, cache)
          .map<std::uint32_t>("product-window-map", count,
                              [](auto value) { return (value & 3u) + 1u; })
          .window({.op = Window::Sum, .radius = radius})
          .filter([](auto value) { return (value & 7u) != 0u; })
          .reduce(Reduce::Sum);
    });
  };
  ok = exact_window(4096u, 4u, "n4096_r4") && ok;
  ok = exact_window(4096u, 1024u, "n4096_r1024") && ok;
  ok = exact_window(BoundedCapacity, 4u, "n262144_r4") && ok;
  ok = exact_window(BoundedCapacity, 1024u, "n262144_r1024") && ok;

  const auto exact_pool = [&](const std::size_t count, const std::size_t width,
                              const std::size_t stride,
                              const std::string_view name) {
    std::vector<std::uint32_t> input(count);
    fill_input(input, count);
    const auto mapped = mapped_prefix(input, count);
    const std::uint32_t expected = pool_sum_oracle(mapped, width, stride);
    const ProductShape shape{.chain = "pool",
                             .name = name,
                             .capacity = count,
                             .active = count,
                             .radius = 0u,
                             .window = width,
                             .stride = stride};
    return measure_exact(device, cache, backend, shape, input, expected, [&] {
      return on(device, cache)
          .map<std::uint32_t>("product-pool-map", count,
                              [](auto value) { return (value & 3u) + 1u; })
          .pool({.op = Window::Sum,
                 .width = width,
                 .stride = stride,
                 .edge = WindowEdge::Clip,
                 .tail = PoolTail::Keep})
          .filter([](auto value) { return (value & 7u) != 0u; })
          .reduce(Reduce::Sum);
    });
  };
  ok = exact_pool(4096u, 129u, 2u, "n4096_k129_s2") && ok;
  ok = exact_pool(BoundedCapacity, 2049u, 2u, "n262144_k2049_s2") && ok;
  ok = measure_bounded(device, cache, backend) && ok;
  return ok;
}

} // namespace rund::measure::compute
