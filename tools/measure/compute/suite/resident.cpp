#include "core.hpp"

namespace rund::measure::compute {

bool ResidentSetup(const Backend backend, const std::size_t count,
                   const std::size_t iterations) {
  std::vector<std::int32_t> input(count);
  for (std::size_t index = 0u; index < count; ++index) {
    input[index] = static_cast<std::int32_t>(index * 2u + 1u);
  }
  auto program =
      rund::compute::on(TargetFor(backend))
          .map<std::int32_t>("resident-setup", count,
                             [](auto value) { return value * 2 + 3; })
          .compile();
  if (!program) {
    std::fprintf(stderr, "resident setup %s compile failed: %.*s\n",
                 Name(backend), static_cast<int>(program.error().size()),
                 program.error().data());
    return false;
  }

  const std::uint64_t input_bytes =
      static_cast<std::uint64_t>(count) * sizeof(std::int32_t);
  const std::uint64_t resident_bytes = input_bytes * 3u;
  std::vector<double> samples{};
  samples.reserve(iterations);
  HashEvidence evidence{};
  for (std::size_t sample = 0u; sample < iterations; ++sample) {
    const auto begin = Clock::now();
    auto job = program->resident(input);
    const auto end = Clock::now();
    if (!job) {
      std::fprintf(stderr, "resident setup %s failed: %.*s\n", Name(backend),
                   static_cast<int>(job.error().size()), job.error().data());
      return false;
    }
    const auto memory = job->memory();
    if (memory.transfer.current != 0u || memory.transfer.peak != input_bytes ||
        memory.transfer.cumulative != input_bytes ||
        memory.resident.current != resident_bytes ||
        memory.resident.budget != resident_bytes) {
      std::fprintf(
          stderr,
          "resident setup %s byte contract failed: "
          "transfer=%llu/%llu/%llu resident=%llu/%llu expected=%llu/%llu\n",
          Name(backend),
          static_cast<unsigned long long>(memory.transfer.current),
          static_cast<unsigned long long>(memory.transfer.peak),
          static_cast<unsigned long long>(memory.transfer.cumulative),
          static_cast<unsigned long long>(memory.resident.current),
          static_cast<unsigned long long>(memory.resident.budget),
          static_cast<unsigned long long>(input_bytes),
          static_cast<unsigned long long>(resident_bytes));
      return false;
    }
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());

    if (sample != 0u) {
      continue;
    }
    if (!job->run()) {
      std::fprintf(stderr, "resident setup %s first run failed\n",
                   Name(backend));
      return false;
    }
    const auto output = job->read();
    if (!output || output->size() != count) {
      std::fprintf(stderr, "resident setup %s first read failed\n",
                   Name(backend));
      return false;
    }
    for (std::size_t index = 0u; index < count; ++index) {
      if ((*output)[index] != input[index] * 2 + 3) {
        std::fprintf(stderr, "resident setup %s first output mismatch at %zu\n",
                     Name(backend), index);
        return false;
      }
    }
    const auto stats = job->stats();
    evidence =
        HashEvidence{.graph = stats.graph_hash, .output = stats.output_hash};
  }
  const bool valid = !samples.empty() && evidence.valid();
  std::printf("resident_setup,%s,%s,%zu,%zu,%.3f,%llu,%llu,%llu,%llu\n",
              Name(backend), valid ? "ok" : "evidence_failed", count,
              iterations, Median(samples),
              static_cast<unsigned long long>(input_bytes),
              static_cast<unsigned long long>(resident_bytes),
              static_cast<unsigned long long>(evidence.graph),
              static_cast<unsigned long long>(evidence.output));
  return valid;
}


} // namespace rund::measure::compute
