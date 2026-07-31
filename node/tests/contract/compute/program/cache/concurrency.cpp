#include "concurrency/local.hpp"

#include <rund/compute.hpp>

int RunComputeProgramCacheConcurrencyContract() {
  using namespace node_compute_cache_contract;
  if (const int code = RunService(); code != 0) {
    return code;
  }
  auto device = rund::compute::open(rund::compute::Target::cpu(2u),
                                    {.workers = 2u, .capacity = 4u});
  if (!device) {
    return 8;
  }
  if (const int code = RunAsync(*device); code != 0) {
    return code;
  }
  if (const int code = RunCapacity(); code != 0) {
    return code;
  }
  if (const int code = RunLifetime(); code != 0) {
    return code;
  }
  return RunFailure();
}
