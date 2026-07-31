#include <rund/compute.hpp>

#include "../allocation.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <span>
#include <thread>
#include <vector>

namespace node_compute_cpu {
[[nodiscard]] int CheckViewFootprint() noexcept;
[[nodiscard]] int CheckCanonicalView();
}

namespace {

template <int Terms, class Value>
[[nodiscard]] constexpr auto AddTerms(const Value value) {
  if constexpr (Terms == 0) {
    return value;
  } else {
    return AddTerms<Terms - 1>(value) + value;
  }
}

struct Observation final {
  std::vector<std::int64_t> output{};
  rund::compute::Stats stats{};
};

template <class T>
bool Observe(const std::size_t count, const std::uint32_t workers,
             Observation &observation) {
  auto device = rund::compute::open(rund::compute::Target::cpu(workers));
  if (!device) {
    return false;
  }

  auto program =
      rund::compute::on(device.value())
          .template map<T>("cpu-worker-tiles", count,
                           [](auto value) { return AddTerms<40>(value) + 7; })
          .compile();
  if (!program) {
    return false;
  }

  std::vector<T> input(count);
  for (std::size_t index = 0; index < count; ++index) {
    input[index] = static_cast<T>(index % 97u);
  }
  auto source = device.value().upload(std::span<const T>{input});
  auto target = device.value().buffer<T>(count);
  if (!source || !target) {
    return false;
  }
  auto first = program.value().run(source.value(), target.value());
  if (!first) {
    return false;
  }
  node_compute_allocation::Start();
  auto run = program.value().run(source.value(), target.value());
  node_compute_allocation::Stop();
  const std::uint64_t warm_allocations = node_compute_allocation::Count();
  if (!run || warm_allocations != 0u) {
    std::fprintf(stderr, "cpu warm heap allocations=%llu\n",
                 static_cast<unsigned long long>(warm_allocations));
    return false;
  }

  std::vector<T> output(count);
  if (!run.value().read(target.value(), std::span<T>{output})) {
    return false;
  }
  observation.output.resize(count);
  for (std::size_t index = 0; index < count; ++index) {
    const T expected = static_cast<T>(input[index] * 41 + 7);
    if (output[index] != expected) {
      return false;
    }
    observation.output[index] = static_cast<std::int64_t>(output[index]);
  }
  observation.stats = run.value().stats();
  return observation.stats.graph_hash != 0u &&
         observation.stats.output_hash != 0u;
}

} // namespace

int RunComputeCpuTilesContract() {
  const int view = node_compute_cpu::CheckViewFootprint();
  if (view != 0) {
    return 100 + view;
  }
  const int canonical_view = node_compute_cpu::CheckCanonicalView();
  if (canonical_view != 0) {
    return 200 + canonical_view;
  }
  const unsigned available = std::thread::hardware_concurrency();
  const std::uint32_t workers =
      std::max(1u, std::min(available == 0u ? 1u : available, 4u));
  constexpr std::size_t kProbeCount = 64u * 1024u + 17u;

  Observation one{};
  Observation many{};
  if (!Observe<std::int32_t>(kProbeCount, 1u, one) ||
      !Observe<std::int32_t>(kProbeCount, workers, many)) {
    return 1;
  }
  if (one.output != many.output ||
      one.stats.graph_hash != many.stats.graph_hash ||
      one.stats.output_hash != many.stats.output_hash) {
    return 2;
  }
  if (one.stats.worker_count != 1u || one.stats.participating_workers != 1u ||
      many.stats.worker_count != workers || many.stats.tile_count < 2u ||
      many.stats.tile_size == 0u || many.stats.vector_chunks == 0u ||
      many.stats.tail_chunks == 0u ||
      (workers > 1u && many.stats.participating_workers < 2u)) {
    std::fprintf(stderr,
                 "cpu tile evidence workers=%u participating=%u tiles=%llu "
                 "tile_size=%llu requested=%u\n",
                 many.stats.worker_count, many.stats.participating_workers,
                 static_cast<unsigned long long>(many.stats.tile_count),
                 static_cast<unsigned long long>(many.stats.tile_size),
                 workers);
    return 3;
  }

  const std::uint64_t tile_size = many.stats.tile_size;
  if (tile_size > static_cast<std::uint64_t>(SIZE_MAX) || tile_size < 2u) {
    return 4;
  }
  const std::size_t tile = static_cast<std::size_t>(tile_size);
  const std::size_t counts[]{1u, tile - 1u, tile, tile + 1u,
                             tile * static_cast<std::size_t>(workers + 1u) +
                                 17u};
  for (const std::size_t count : counts) {
    Observation fixed_lane32{};
    Observation fixed_lane64{};
    if (!Observe<std::int32_t>(count, workers, fixed_lane32) ||
        !Observe<std::int64_t>(count, workers, fixed_lane64) ||
        fixed_lane32.output != fixed_lane64.output) {
      return 5;
    }
  }
  return 0;
}
