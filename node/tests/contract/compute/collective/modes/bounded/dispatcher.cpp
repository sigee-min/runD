#include "local.hpp"

namespace rund_node_collective_modes {

bool CheckBounded(const rund::compute::Backend backend,
                  DomainEvidence &evidence, const Domain domain) {
  return bounded::CheckAggregate(backend, evidence, domain) &&
         bounded::CheckWindow(backend, evidence, domain) &&
         bounded::CheckResident(backend, domain);
}

} // namespace rund_node_collective_modes
