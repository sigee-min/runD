#include "reference.hpp"

#include <cstdio>

namespace rund::measure::compute {

ReferenceLedger<ReferenceCapacity> references{};
HashEvidence batch_reference{};

bool CheckReference(const Backend backend, const ReferenceKey key,
                    const HashEvidence evidence) {
  if (!key.valid()) {
    return true;
  }
  const ReferenceStatus status =
      references.check(backend == Backend::Cpu, key, evidence);
  if (status == ReferenceStatus::Established ||
      status == ReferenceStatus::Matched) {
    return true;
  }
  std::fprintf(stderr,
               "reference %s/%.*s/%.*s/%zu failed: status=%u graph=%llu "
               "output=%llu\n",
               Name(backend), static_cast<int>(key.workload.size()),
               key.workload.data(), static_cast<int>(key.variant.size()),
               key.variant.data(), key.count, static_cast<unsigned>(status),
               static_cast<unsigned long long>(evidence.graph),
               static_cast<unsigned long long>(evidence.output));
  return false;
}

} // namespace rund::measure::compute
