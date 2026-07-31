#include "local.hpp"

#include "../../../allocation.hpp"
#include <rund/compute/async.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace node_compute_cache_contract {
namespace {

[[nodiscard]] bool async_compile_coalesces(rund::compute::Device &device) {
  auto cache = rund::compute::program_cache(device, 4u);
  if (!cache) {
    return false;
  }
  auto first_pending =
      rund::compute::on(device, *cache)
          .map<std::int32_t>("async-first", 4u,
                             [](auto value) { return value * 3; })
          .compile_async();
  auto second_pending =
      rund::compute::on(device, *cache)
          .map<std::int32_t>("async-renamed", 4u,
                             [](auto value) { return value * 3; })
          .compile_async();
  if (!first_pending || !second_pending) {
    return false;
  }
  auto first = first_pending->get();
  auto second = second_pending->get();
  const auto stats = cache->stats();
  if (!first || !second || first->fingerprint() != second->fingerprint() ||
      stats.misses != 1u || stats.hits + stats.waits != 1u ||
      stats.in_flight != 0u) {
    return false;
  }
  constexpr std::array<std::int32_t, 4u> input{1, 2, 3, 4};
  auto output = first->run(std::span<const std::int32_t>{input});
  return output && *output == std::vector<std::int32_t>{3, 6, 9, 12};
}

[[nodiscard]] bool
async_allocation_failure_releases_capacity(rund::compute::Device &device) {
  auto failed_flow = rund::compute::on(device).map<std::int32_t>(
      "async-allocation-failure", 4u, [](auto value) { return value + 1; });
  auto recovered_flow = rund::compute::on(device).map<std::int32_t>(
      "async-allocation-recovery", 4u, [](auto value) { return value + 2; });
  node_compute_allocation::FailNext();
  auto failed = std::move(failed_flow).compile_async();
  if (failed ||
      failed.reason() != rund::compute::Reason::AsyncCompileUnavailable) {
    return false;
  }
  auto recovered = std::move(recovered_flow).compile_async();
  if (!recovered) {
    return false;
  }
  auto program = recovered->get();
  return static_cast<bool>(program);
}

} // namespace

int RunAsync(rund::compute::Device &device) {
  if (!async_compile_coalesces(device)) {
    return 9;
  }
  if (!async_allocation_failure_releases_capacity(device)) {
    return 10;
  }
  return 0;
}

} // namespace node_compute_cache_contract
