#pragma once

#include "base.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

inline constexpr std::uint32_t VulkanResidencyAdmissionNoOrdinal =
    std::numeric_limits<std::uint32_t>::max();

enum class VulkanResidencyAdmissionCandidate : std::uint8_t {
  GraphDirect,
  GraphStageGeneratedIndirect,
  GraphStageSequence,
  GeneratedIndirect,
  Direct,
};

enum class VulkanResidencyAdmissionReason : std::uint8_t {
  NotEvaluated,
  InvalidPipeline,
  PredicateRejected,
  PredicateAccepted,
};

enum class VulkanResidencyGraphGeneratedPredicate : std::uint8_t {
  None,
  Pipeline,
  EntryIdentity,
  EntryShape,
  BindingIndex,
  BindingRole,
  BindingAlias,
  ControlBinding,
  Semantic,
};

struct VulkanResidencyAdmissionCandidateSnapshot final {
  VulkanResidencyAdmissionCandidate candidate{
      VulkanResidencyAdmissionCandidate::GraphDirect};
  VulkanResidencyAdmissionReason reason{
      VulkanResidencyAdmissionReason::NotEvaluated};
  std::uint32_t entry_ordinal{VulkanResidencyAdmissionNoOrdinal};
  std::uint32_t local_ordinal{VulkanResidencyAdmissionNoOrdinal};
  std::uint32_t predicate_ordinal{VulkanResidencyAdmissionNoOrdinal};
  std::uint32_t entry_count{};
  std::uint32_t active_step_count{};
  std::uint32_t data_binding_count{};
  std::uint32_t control_binding_count{};
  std::uint64_t generation_tag{};
  VulkanResidencyGraphGeneratedPredicate first_failure_predicate{
      VulkanResidencyGraphGeneratedPredicate::None};
  std::uint32_t first_failure_ordinal{VulkanResidencyAdmissionNoOrdinal};
  const char *first_failure_reason{"not_evaluated"};
};

struct VulkanResidencyAdmissionSnapshot final {
  std::array<VulkanResidencyAdmissionCandidateSnapshot, 5u> candidates{};
  std::uint32_t candidate_count{};
  VulkanResidencyAdmissionCandidate final_candidate{
      VulkanResidencyAdmissionCandidate::GraphDirect};
  VulkanResidencyMode selected_mode{VulkanResidencyMode::Direct};
  std::uint64_t generation_tag{};
  bool final_selected{};
};

enum class VulkanResidencyGraphGeneratedPhase : std::uint8_t {
  NotApplicable,
  Admitted,
  Recorded,
  Submitted,
  TerminalKnown,
  TerminalUnknown,
  Quarantined,
};

struct VulkanResidencyGraphGeneratedDiagnostics final {
  VulkanResidencyGraphGeneratedPhase phase{
      VulkanResidencyGraphGeneratedPhase::NotApplicable};
  const char *first_reason{"not_applicable"};
  std::uint64_t generation{};
  std::uint64_t record_generation{};
  std::uint64_t submission_generation{};
  std::uint32_t admission_count{};
  std::uint32_t ready_count{};
  std::uint32_t recorded_count{};
  std::uint32_t submitted_count{};
  std::uint32_t terminal_count{};
  std::uint32_t accepted_count{};
  std::uint32_t known_failure_count{};
  std::uint32_t unknown_count{};
  std::uint64_t submit_seq{};
  std::uint64_t done_seq{};
  std::uint64_t submit_total{};
  std::uint64_t done_total{};
  std::uint64_t accept_total{};
  std::uint64_t known_total{};
  std::uint64_t unknown_total{};
  VulkanResidencyGraphGeneratedPredicate first_failure_predicate{
      VulkanResidencyGraphGeneratedPredicate::None};
  std::uint32_t first_failure_ordinal{VulkanResidencyAdmissionNoOrdinal};
  bool replay{};
  bool generation_overflow{};
  bool failed{};
  bool quarantined{};
};

#endif

} // namespace rund::node::accel::detail
