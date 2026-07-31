#include "model.hpp"

#include <utility>

namespace rund_node_graph_services {

[[nodiscard]] int CheckCache(rund::compute::Device *const device) {
  auto one = rund::compute::program_cache(*device, 1u);
  if (!one || !build(*device, *one, "one", 1) ||
      !build(*device, *one, "two", 2) ||
      !build(*device, *one, "one-again", 1)) {
    return 14;
  }
  const auto evicted = one->stats();
  if (evicted.misses != 3u || evicted.evictions != 2u ||
      evicted.ready_entries != 1u) {
    return 15;
  }
  one->clear();
  if (one->stats().ready_entries != 0u) {
    return 16;
  }
  if (!build(*device, *one, "after-clear", 1) || one->stats().misses != 4u) {
    return 17;
  }

  auto zero = rund::compute::program_cache(*device, 0u);
  if (zero || zero.error() != "compute_program_cache_capacity") {
    return 18;
  }
  auto moved = rund::compute::program_cache(*device, 1u);
  if (!moved) {
    return 19;
  }
  auto live = std::move(*moved);
  auto invalid = rund::compute::on(*device, *moved)
                     .map<std::int32_t>("moved-cache", 4u,
                                        [](auto value) { return value; })
                     .compile();
  if (invalid || invalid.error() != "compute_program_cache_invalid" ||
      live.stats().capacity != 1u) {
    return 20;
  }
  return 0;
}

} // namespace rund_node_graph_services
