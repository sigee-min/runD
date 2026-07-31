#include <rund/compute.hpp>
#include <rund/compute/async.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <future>
#include <span>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

template <class Flow>
auto Compile(Flow flow, const char *name, const std::size_t count) {
  return std::move(flow)
      .template map<std::int32_t>(name, count,
                                  [](auto value) {
                                    const auto a = value * 3 + 7;
                                    const auto b = a * 5 - value * 2;
                                    return b * 7 + a;
                                  })
      .compile();
}

template <class Function> double MedianMicros(Function function) {
  std::array<double, 21u> samples{};
  for (double &sample : samples) {
    const auto begin = Clock::now();
    if (!function()) {
      return -1.0;
    }
    const auto end = Clock::now();
    sample = std::chrono::duration<double, std::micro>(end - begin).count();
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2u];
}

struct InternalMemory final {
  bool complete = false;
  std::size_t count{};
  std::uint64_t bytes{};
};

template <class Owner>
[[nodiscard]] InternalMemory PhysicalInternalMemory(const Owner &owner) {
  std::array<rund::compute::MemoryEntry, 64u> entries{};
  const rund::compute::MemorySnapshot snapshot = owner.memory_snapshot(entries);
  InternalMemory result{.complete = !snapshot.truncated()};
  for (std::size_t index = 0u; index < snapshot.written; ++index) {
    const rund::compute::MemoryEntry &entry = entries[index];
    if (entry.use != rund::compute::MemoryUse::Internal ||
        (entry.category != rund::compute::MemoryCategory::Host &&
         entry.category != rund::compute::MemoryCategory::Device)) {
      continue;
    }
    ++result.count;
    result.bytes += entry.bytes.current;
  }
  return result;
}

bool Cache(rund::compute::Device &device) {
  constexpr std::size_t count = 4096u;
  constexpr std::size_t repeats = 101u;
  const auto uncached_begin = Clock::now();
  for (std::size_t index = 0u; index < repeats; ++index) {
    auto program = Compile(rund::compute::on(device), "uncached", count);
    if (!program) {
      return false;
    }
  }
  const auto uncached_end = Clock::now();

  auto cache = rund::compute::program_cache(device, 4u);
  if (!cache) {
    return false;
  }
  const auto cached_begin = Clock::now();
  for (std::size_t index = 0u; index < repeats; ++index) {
    auto program = Compile(rund::compute::on(device, *cache), "cached", count);
    if (!program) {
      return false;
    }
  }
  const auto cached_end = Clock::now();
  const auto stats = cache->stats();
  if (stats.misses != 1u || stats.hits != repeats - 1u) {
    return false;
  }
  const double uncached_us =
      std::chrono::duration<double, std::micro>(uncached_end - uncached_begin)
          .count();
  const double cached_us =
      std::chrono::duration<double, std::micro>(cached_end - cached_begin)
          .count();
  std::printf("program_cache_uncached\t%zu\tus\t%.3f\tcompiles\t%zu\n", repeats,
              uncached_us, repeats);
  std::printf("program_cache_cached\t%zu\tus\t%.3f\tcache_misses\t%llu\n",
              repeats, cached_us,
              static_cast<unsigned long long>(stats.misses));
  std::printf("program_cache_speedup\t%zu\tratio\t%.3f\n", repeats,
              uncached_us / cached_us);
  return true;
}

bool Coalescing(rund::compute::Device &device) {
  constexpr std::size_t requests = 32u;
  auto cache = rund::compute::program_cache(device, 4u);
  if (!cache) {
    return false;
  }
  using ProgramResult =
      rund::compute::Result<rund::compute::Program<std::int32_t(std::int32_t)>>;
  std::vector<std::future<ProgramResult>> pending;
  pending.reserve(requests);
  const auto begin = Clock::now();
  for (std::size_t index = 0u; index < requests; ++index) {
    auto request =
        rund::compute::on(device, *cache)
            .map<std::int32_t>("coalesced", 4096u,
                               [](auto value) { return value * 9 + 1; })
            .compile_async();
    if (!request) {
      return false;
    }
    pending.push_back(std::move(*request));
  }
  for (auto &request : pending) {
    if (!request.get()) {
      return false;
    }
  }
  const auto end = Clock::now();
  const auto stats = cache->stats();
  if (stats.misses != 1u || stats.hits + stats.waits != requests - 1u) {
    return false;
  }
  const double elapsed =
      std::chrono::duration<double, std::micro>(end - begin).count();
  std::printf("async_same_key\t%zu\tus\t%.3f\tcache_misses\t%llu\twaits\t%"
              "llu\thits\t%llu\n",
              requests, elapsed, static_cast<unsigned long long>(stats.misses),
              static_cast<unsigned long long>(stats.waits),
              static_cast<unsigned long long>(stats.hits));
  return true;
}

bool GraphStorage(rund::compute::Device &device) {
  constexpr std::size_t count = 1u << 18u;
  constexpr std::size_t external_values = 2u;
  constexpr std::size_t input_values = 1u;
  constexpr std::size_t output_values = external_values - input_values;
  constexpr std::size_t internal_values = 1u;
  constexpr std::size_t concurrent_jobs = 8u;
  constexpr std::uint64_t payload_bytes = count * sizeof(std::uint32_t);
  constexpr std::uint64_t internal_bytes = internal_values * payload_bytes;
  constexpr std::uint64_t job_bytes =
      (2u * input_values + output_values + internal_values) * payload_bytes;
  constexpr std::uint64_t expected_bytes = concurrent_jobs * job_bytes;
  std::vector<std::uint32_t> input(count);
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] = static_cast<std::uint32_t>(index & 15u);
  }
  const std::uint64_t baseline = device.memory().host.current;
  std::uint64_t observed_bytes = 0u;
  double setup_us = 0.0;
  {
    auto program =
        rund::compute::on(device)
            .input<std::uint32_t>(count)
            .map("graph-storage-map", [](auto value) { return value + 1u; })
            .scan(rund::compute::Scan::InclusiveSum)
            .compile();
    if (!program) {
      return false;
    }
    const InternalMemory program_internal = PhysicalInternalMemory(*program);
    if (!program_internal.complete || program_internal.count != 0u ||
        program_internal.bytes != 0u ||
        device.memory().host.current != baseline) {
      return false;
    }
    std::vector<rund::compute::Job<std::uint32_t(std::uint32_t)>> residents;
    residents.reserve(concurrent_jobs);
    const auto begin = Clock::now();
    for (std::size_t index = 0u; index < concurrent_jobs; ++index) {
      auto resident = program->resident(std::span<const std::uint32_t>{input});
      if (!resident || resident->memory().resident.current != job_bytes) {
        return false;
      }
      const InternalMemory internal = PhysicalInternalMemory(*resident);
      if (!internal.complete || internal.count != internal_values ||
          internal.bytes != internal_bytes) {
        return false;
      }
      residents.push_back(std::move(*resident));
    }
    const auto end = Clock::now();
    setup_us = std::chrono::duration<double, std::micro>(end - begin).count();
    const std::uint64_t current = device.memory().host.current;
    if (current < baseline) {
      return false;
    }
    observed_bytes = current - baseline;
    if (observed_bytes != expected_bytes || !residents.front().run()) {
      return false;
    }
    const auto output = residents.front().read();
    const rund::compute::Stats stats = residents.front().stats();
    if (!output || output->size() != count || output->front() != 1u ||
        stats.graph_hash == 0u || stats.output_hash == 0u) {
      return false;
    }
  }
  if (device.memory().host.current != baseline) {
    return false;
  }
  std::printf("graph_storage\t%zu\tvalues_E\t%zu\tvalues_K\t%zu\tjobs_C\t"
              "%zu\tpayload_bytes\t%llu\texpected_bytes\t%llu\t"
              "observed_bytes\t%llu\tsetup_us\t%.3f\n",
              count, external_values, internal_values, concurrent_jobs,
              static_cast<unsigned long long>(payload_bytes),
              static_cast<unsigned long long>(expected_bytes),
              static_cast<unsigned long long>(observed_bytes), setup_us);
  return true;
}

bool Bounded(rund::compute::Device &device) {
  constexpr std::size_t count = 1u << 18u;
  std::vector<std::uint32_t> input(count);
  for (std::size_t index = 0u; index < count; ++index) {
    input[index] = static_cast<std::uint32_t>(index);
  }
  const auto expensive = [](auto value) {
    const auto a = (value * 1664525u + 1013904223u) ^
                   rund::compute::shr_logical<7u>(value);
    const auto b = (a * 22695477u + 1u) ^ rund::compute::shr_logical<9u>(a);
    const auto c =
        (b * 1103515245u + 12345u) ^ rund::compute::shr_logical<11u>(b);
    const auto d =
        (c * 214013u + 2531011u) ^ rund::compute::shr_logical<13u>(c);
    return (d * 134775813u + 1u) ^ rund::compute::shr_logical<15u>(d);
  };
  auto dense_program =
      rund::compute::on(device)
          .map<std::uint32_t>("dense-expensive", count, expensive)
          .compile();
  auto sparse_program =
      rund::compute::on(device)
          .map<std::uint32_t>("bounded-source", count,
                              [](auto value) { return value; })
          .filter([](auto value) { return (value & 15) == 0; })
          .map("bounded-expensive", expensive)
          .compile();
  if (!dense_program || !sparse_program) {
    return false;
  }
  auto dense = dense_program->resident(std::span<const std::uint32_t>{input});
  auto sparse = sparse_program->resident(std::span<const std::uint32_t>{input});
  if (!dense || !sparse || !dense->run() || !sparse->run()) {
    return false;
  }
  auto sparse_values = sparse->read();
  if (!sparse_values || sparse_values->size() != count / 16u) {
    return false;
  }
  const double dense_us = MedianMicros([&] { return bool(dense->run()); });
  const double sparse_us = MedianMicros([&] { return bool(sparse->run()); });
  if (dense_us <= 0.0 || sparse_us <= 0.0) {
    return false;
  }
  std::printf("bounded_map_dense\t%zu\tus\t%.3f\tactive\t%zu\n", count,
              dense_us, count);
  std::printf("bounded_map_sparse\t%zu\tus\t%.3f\tactive\t%zu\n", count,
              sparse_us, count / 16u);
  std::printf("bounded_map_sparse_ratio\t%zu\tratio\t%.3f\n", count,
              dense_us / sparse_us);
  return true;
}

bool BoundedSort(rund::compute::Device &device) {
  constexpr std::size_t count = 1u << 18u;
  std::vector<std::uint32_t> input(count);
  for (std::size_t index = 0u; index < count; ++index) {
    input[index] = static_cast<std::uint32_t>(count - index);
  }
  auto dense_program = rund::compute::on(device)
                           .map<std::uint32_t>("dense-sort-source", count,
                                               [](auto value) { return value; })
                           .sort()
                           .compile();
  auto sparse_program =
      rund::compute::on(device)
          .map<std::uint32_t>("bounded-sort-source", count,
                              [](auto value) { return value; })
          .filter([](auto value) { return (value & 15u) == 0u; })
          .sort()
          .compile();
  if (!dense_program || !sparse_program) {
    return false;
  }
  auto dense = dense_program->resident(std::span<const std::uint32_t>{input});
  auto sparse = sparse_program->resident(std::span<const std::uint32_t>{input});
  if (!dense || !sparse || !dense->run() || !sparse->run()) {
    return false;
  }
  auto sparse_values = sparse->read();
  if (!sparse_values || sparse_values->size() != count / 16u ||
      !std::is_sorted(sparse_values->begin(), sparse_values->end())) {
    return false;
  }
  const double dense_us = MedianMicros([&] { return bool(dense->run()); });
  const double sparse_us = MedianMicros([&] { return bool(sparse->run()); });
  if (dense_us <= 0.0 || sparse_us <= 0.0) {
    return false;
  }
  std::printf("bounded_sort_dense\t%zu\tus\t%.3f\tactive\t%zu\n", count,
              dense_us, count);
  std::printf("bounded_sort_sparse\t%zu\tus\t%.3f\tactive\t%zu\n", count,
              sparse_us, count / 16u);
  std::printf("bounded_sort_sparse_speedup\t%zu\tratio\t%.3f\n", count,
              dense_us / sparse_us);
  return true;
}

} // namespace

int main() {
  auto device = rund::compute::open(
      rund::compute::Target::cpu(1u),
      rund::compute::Compile{.workers = 2u, .capacity = 64u});
  if (!device) {
    return 1;
  }
  if (!Cache(*device)) {
    return 2;
  }
  if (!Coalescing(*device)) {
    return 3;
  }
  if (!GraphStorage(*device)) {
    return 4;
  }
  if (!Bounded(*device)) {
    return 5;
  }
  if (!BoundedSort(*device)) {
    return 6;
  }
  return 0;
}
