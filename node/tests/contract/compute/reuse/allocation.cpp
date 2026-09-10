#include "local.hpp"

#include "../allocation.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace rund_node_test_compute_reuse {
namespace {

[[nodiscard]] std::uint64_t HashBytes(const void *const data,
                                      const std::size_t bytes) noexcept {
  constexpr std::uint64_t offset = 1469598103934665603ull;
  constexpr std::uint64_t prime = 1099511628211ull;
  const auto *const values = static_cast<const std::uint8_t *>(data);
  std::uint64_t hash = offset;
  for (std::size_t index = 0u; index < bytes; ++index) {
    hash ^= values[index];
    hash *= prime;
  }
  return hash;
}

} // namespace

[[nodiscard]] bool
CheckInitialHostReuse(ReuseProgram &program,
                      const std::span<const std::int32_t> input) {
  auto warm_host = program.run(input);
  if (!warm_host) {
    return false;
  }
  node_compute_allocation::Start();
  auto host = program.run(input);
  node_compute_allocation::Stop();
  const std::uint64_t host_allocations = node_compute_allocation::Count();
  if (!host || host_allocations != 1u ||
      *host != std::vector<std::int32_t>{7, 9, 11, 13}) {
    std::fprintf(stderr, "host one-shot heap allocations=%llu\n",
                 static_cast<unsigned long long>(host_allocations));
    return false;
  }
  return true;
}

[[nodiscard]] int
CheckBufferReuseAndReceipt(rund::compute::Device &device, ReuseProgram &program,
                           const std::span<const std::int32_t> input) {
  auto source = device.upload(input);
  auto target = device.buffer<std::int32_t>(input.size());
  if (!source || !target) {
    return 3;
  }

  auto first = program.run(source.value(), target.value());
  node_compute_allocation::Start();
  auto second = program.run(source.value(), target.value());
  node_compute_allocation::Stop();
  if (!first || !second) {
    return 4;
  }
  if (node_compute_allocation::Count() != 0u) {
    std::fprintf(
        stderr, "reuse warm heap allocations=%llu\n",
        static_cast<unsigned long long>(node_compute_allocation::Count()));
    return 5;
  }

  node_compute_allocation::Start();
  rund::compute::Run copied = second.value();
  rund::compute::Run moved = std::move(copied);
  node_compute_allocation::Stop();
  if (node_compute_allocation::Count() != 0u ||
      moved.stats().graph_hash != second.value().stats().graph_hash) {
    std::fprintf(
        stderr, "run receipt copy heap allocations=%llu\n",
        static_cast<unsigned long long>(node_compute_allocation::Count()));
    return 5;
  }
  std::array<std::int32_t, 4> copied_output{};
  if (!moved.read(target.value(), std::span<std::int32_t>{copied_output}) ||
      moved.stats().download_events != 1u ||
      second.value().stats().download_events != 0u ||
      copied_output != std::array<std::int32_t, 4>{7, 9, 11, 13}) {
    std::fprintf(stderr, "run receipt telemetry copy did not diverge\n");
    return 5;
  }

  const rund::compute::Stats warm = second.value().stats();
  if (warm.backend != rund::compute::Backend::Cpu ||
      warm.pipeline_compiles != 0 || warm.buffer_allocations != 0 ||
      warm.download_events != 0) {
    std::fprintf(stderr,
                 "warm stats pipeline_compile=%llu buffer_allocation=%llu "
                 "download_event=%llu\n",
                 static_cast<unsigned long long>(warm.pipeline_compiles),
                 static_cast<unsigned long long>(warm.buffer_allocations),
                 static_cast<unsigned long long>(warm.download_events));
    return 6;
  }
  if (warm.dispatches != 1 || warm.graph_hash == 0 || warm.output_hash != 0) {
    std::fprintf(stderr, "warm identity submit=%llu graph=%llu output=%llu\n",
                 static_cast<unsigned long long>(warm.dispatches),
                 static_cast<unsigned long long>(warm.graph_hash),
                 static_cast<unsigned long long>(warm.output_hash));
    return 7;
  }

  std::array<std::int32_t, 4> output{};
  if (!second.value().read(target.value(), std::span<std::int32_t>{output})) {
    return 8;
  }
  const rund::compute::Stats read = second.value().stats();
  if (read.download_events != 1 || read.downloaded_bytes != sizeof(output) ||
      read.output_hash != HashBytes(output.data(), sizeof(output)) ||
      output != std::array<std::int32_t, 4>{7, 9, 11, 13}) {
    std::fprintf(stderr,
                 "read stats download_events=%llu downloaded_bytes=%llu "
                 "output=%llu\n",
                 static_cast<unsigned long long>(read.download_events),
                 static_cast<unsigned long long>(read.downloaded_bytes),
                 static_cast<unsigned long long>(read.output_hash));
    return 9;
  }
  return 0;
}

} // namespace rund_node_test_compute_reuse
