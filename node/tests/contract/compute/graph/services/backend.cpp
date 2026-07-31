#include "model.hpp"

#include "../../../target/selection.hpp"

#include <array>
#include <cstdio>
#include <vector>

namespace rund_node_graph_services {

[[nodiscard]] int CheckBackends() {
  const std::array<std::int32_t, 4> low{1, 2, 3, 4};
  rund::compute::graph::Fingerprint backend_fingerprint{};
  CrossBackendReferences cross_backend_references{};
  for (const Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    auto backend_device = rund::compute::open(
        rund::node::test_contract::target_for(backend),
        rund::compute::Compile{.workers = 2u, .capacity = 4u});
    if (!backend_device) {
      const std::string_view error = backend_device.error();
      std::fprintf(stderr, "graph-services backend=%u open failed: %.*s\n",
                   static_cast<unsigned>(backend),
                   static_cast<int>(error.size()),
                   error.empty() ? "" : error.data());
      return 24;
    }
    auto backend_cache = rund::compute::program_cache(*backend_device, 2u);
    if (!backend_cache) {
      return 25;
    }
    auto backend_first =
        build(*backend_device, *backend_cache, "backend-first", 7);
    auto backend_second =
        build(*backend_device, *backend_cache, "backend-renamed", 7);
    if (!backend_first || !backend_second ||
        backend_first->fingerprint() != backend_second->fingerprint() ||
        backend_cache->stats().misses != 1u ||
        backend_cache->stats().hits != 1u ||
        !ValidResourceGraph(backend_first->graph())) {
      return 26;
    }
    if (!backend_fingerprint) {
      backend_fingerprint = backend_first->fingerprint();
    } else if (backend_fingerprint != backend_first->fingerprint()) {
      return 27;
    }
    auto backend_output =
        backend_first->run(std::span<const std::int32_t>{low});
    if (!backend_output ||
        *backend_output != std::vector<std::int32_t>{8, 9, 10, 11}) {
      const std::string_view error = backend_output.error();
      std::fprintf(stderr, "graph-services backend=%u run failed: %.*s\n",
                   static_cast<unsigned>(backend),
                   static_cast<int>(error.size()),
                   error.empty() ? "" : error.data());
      return 28;
    }
    if (!CheckMemoryReuse(*backend_device, backend,
                          cross_backend_references.memory)) {
      std::fprintf(stderr, "graph services memory failure backend=%u\n",
                   static_cast<unsigned>(backend));
      return 76;
    }
    if (!CheckAsynchronous(*backend_device, backend,
                           cross_backend_references.asynchronous)) {
      std::fprintf(stderr, "graph services async failure backend=%u\n",
                   static_cast<unsigned>(backend));
      return 75;
    }
    if (!CheckMixedFixed(*backend_device, backend,
                         cross_backend_references.mixed_fixed)) {
      std::fprintf(stderr, "graph services mixed fixed failure backend=%u\n",
                   static_cast<unsigned>(backend));
      return 75;
    }
    if (!CheckFixedCache(*backend_device, backend, cross_backend_references)) {
      std::fprintf(stderr, "graph services cache identity failure backend=%u\n",
                   static_cast<unsigned>(backend));
      return 75;
    }
  }
  return 0;
}

} // namespace rund_node_graph_services
