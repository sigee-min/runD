#include "local.hpp"

#include <cstdio>

namespace rund_node_test_pipeline_residency {

int CheckNativeSliding() {
  using namespace native_sliding;
  const auto check = [](const char *const name, const bool result) {
    if (!result) {
      std::fprintf(stderr, "native sliding failed: %s\n", name);
      return false;
    }
    return true;
  };
  if (!check("admission-reject-no-allocation",
             AdmissionRejectNoAllocationCase()) ||
      !check("q5", SlidingCase(5u)) || !check("q9", SlidingCase(9u)) ||
      !check("q257", SlidingCase(257u)) || !check("stride1", StrideCase(1u)) ||
      !check("stride2", StrideCase(2u)) || !check("stride3", StrideCase(3u)) ||
      !check("project-fail", ProjectionFailureCase(false)) ||
      !check("sibling-project-fail", ProjectionFailureCase(true)) ||
      !check("invalid-mask", InvalidMaskCase()) ||
      !check("inline-reject", InlineContradictionCase()) ||
      !check("cross-thread-early-reject",
             CrossThreadEarlyContradictionCase()) ||
      !check("active-self-retain", ActiveSelfRetainCase()) ||
      !check("reentrant-final", ReentrantFinalCase()) ||
      !check("project-suppression", SuppressionCase(SuppressionStage::Project)) ||
      !check("seed-suppression", SuppressionCase(SuppressionStage::Seed)) ||
      !check("delayed-submit-return", DelayedSubmitReturnCase()) ||
      !check("release-handoff", ReleaseHandoffCase()) ||
      !check("returned-pressure", ReturnedPressureCase()) ||
      !check("returned-progress-rearm", ReturnedProgressRearmCase()) ||
      !check("cross-slot-inline", CrossSlotInlineCase()) ||
      !check("wrong-control-generation", WrongControlGenerationCase()) ||
      !check("nested-inline", NestedInlineCase()) ||
      !check("reverse-failure", ReverseFailureCase()) ||
      !check("multi-reentrant-wake", MultiReentrantWakeCase())) {
    return 1;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency
