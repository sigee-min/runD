#include "local.hpp"

#if defined(RUND_NODE_TEST_BACKEND_CPU)

namespace rund_node_test_persistent_product {

int CheckPersistentProduct(const rund::compute::Backend,
                           const NativeQueueCounter) noexcept {
  return 0;
}

} // namespace rund_node_test_persistent_product

#else

#include <array>
#include <cstdio>

namespace rund_node_test_persistent_product {
namespace {

[[nodiscard]] int ReportFailure(const char *const name) noexcept {
  std::fprintf(stderr, "persistent product subcase failed: %s\n", name);
  return 1;
}

} // namespace

int CheckPersistentProduct(const rund::compute::Backend backend,
                           const NativeQueueCounter queue_counter) noexcept {
  constexpr std::array<std::uint64_t, 3u> coordinates{{2u, 3u, 5u}};
  constexpr std::array<const char *, 3u> coordinate_names{
      {"persistent-q2", "persistent-q3", "persistent-q5"}};
  for (std::size_t index = 0u; index < coordinates.size(); ++index) {
    bool unavailable = false;
    if (!RunPersistentProductCase(backend, queue_counter, coordinates[index],
                                  unavailable)) {
      return ReportFailure(coordinate_names[index]);
    }
    if (unavailable) {
      return 0;
    }
  }
  bool unavailable = false;
  if (!CheckPersistentWindowProduct(backend, queue_counter, unavailable)) {
    return ReportFailure("persistent-window-fallback");
  }
  if (!CheckPersistentMemoryRetry(backend, unavailable)) {
    return ReportFailure("persistent-memory-retry");
  }
  if (!CheckPersistentCapabilityTaxonomy()) {
    return ReportFailure("persistent-capability-query");
  }
  if (!CheckPersistentStartFailure(backend, queue_counter, unavailable)) {
    return ReportFailure("persistent-start-failure");
  }
  if (!CheckPersistentTerminalUnsupported(backend, queue_counter,
                                          unavailable)) {
    return ReportFailure("persistent-terminal-unsupported");
  }
  if (!CheckPersistentKnownAdmissionFailure(backend, queue_counter,
                                            unavailable)) {
    return ReportFailure("persistent-known-admission-failure");
  }
  if (!CheckPersistentPublicationFailure(backend, queue_counter, unavailable)) {
    return ReportFailure("persistent-publication-failure");
  }
  if (!CheckPersistentPublicationObservers(backend, unavailable)) {
    return ReportFailure("persistent-publication-observers");
  }
  if (!CheckPersistentUnknownFailure(backend, queue_counter, unavailable)) {
    return ReportFailure("persistent-unknown-failure");
  }
  return 0;
}

} // namespace rund_node_test_persistent_product

#endif
