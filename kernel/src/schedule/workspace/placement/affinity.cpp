#include "local.hpp"

namespace rund::kernel::workspace_placement {

const char* AffinityPlacementReason(const WorkerBackendCapabilities& capabilities) {
  switch (capabilities.affinity_truth_level) {
    case WorkerTruthLevel::Verified:
      return "verified";
    case WorkerTruthLevel::HintOnly:
      return "affinity_hint_only";
    case WorkerTruthLevel::Unknown:
      return "affinity_truth_unknown";
  }
  return "affinity_truth_unknown";
}

} // namespace rund::kernel::workspace_placement
