#include "services/local.hpp"

int RunComputeGraphServicesContract() {
  using namespace rund_node_graph_services;
  auto device = rund::compute::open(rund::compute::Target::cpu(2u),
                                    {.workers = 2u, .capacity = 4u});
  if (!device) {
    return 1;
  }
  auto cache = rund::compute::program_cache(*device, 4u);
  if (!cache || cache->stats().capacity != 4u) {
    return 2;
  }
  if (const int policy = CheckPolicy(&*device); policy != 0) {
    return policy;
  }
  if (const int identity = CheckIdentity(&*device, &*cache); identity != 0) {
    return identity;
  }
  if (const int bounded = CheckBounded(&*device, &*cache); bounded != 0) {
    return bounded;
  }
  if (const int cache_result = CheckCache(&*device); cache_result != 0) {
    return cache_result;
  }
  if (const int empty = CheckEmpty(&*device); empty != 0) {
    return empty;
  }
  return CheckBackends();
}
