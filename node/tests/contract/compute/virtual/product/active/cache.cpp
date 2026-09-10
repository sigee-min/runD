#include "local.hpp"

namespace rund_node_test_virtual::product::active {

int CheckCache(ActiveFixture &fixture) {
  using namespace rund::compute;

  auto &prepared = *fixture.prepared;
  auto &input_backing = *fixture.input_backing;
  const BackingFacts before_invalidate = input_backing.facts();
  if (!input_backing.invalidate() || !prepared.run(7u)) {
    return 12;
  }

  const Stats invalidated = prepared.stats();
  const BackingFacts after_invalidate = input_backing.facts();
  if (invalidated.pipeline.residency.page_in_count != 1u ||
      invalidated.pipeline.residency.cache_hit_count != 0u ||
      after_invalidate.read_count != before_invalidate.read_count + 1u ||
      after_invalidate.read_bytes !=
          before_invalidate.read_bytes + 7u * sizeof(std::int32_t)) {
    return 13;
  }
  return 0;
}

} // namespace rund_node_test_virtual::product::active
