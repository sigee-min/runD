#include "local.hpp"

namespace node_accel_contract {
namespace {

[[nodiscard]] std::size_t Count(const std::string_view text,
                                const std::string_view needle) {
  std::size_t count = 0u;
  std::size_t at = 0u;
  while ((at = text.find(needle, at)) != std::string_view::npos) {
    ++count;
    at += needle.size();
  }
  return count;
}

[[nodiscard]] bool SourceMatches(const ComputeApi api,
                                 const ComputeScalar scalar,
                                 const bool uniform_invariant = false,
                                 const bool writes_history = false) {
  Fixture fixture{api, scalar, uniform_invariant, writes_history};
  const MapRecurrence recurrence =
      BuildMapRecurrence(fixture.entries, fixture.barriers);
  const auto expected_variant =
      writes_history ? rund::kernel::LoweringArtifactVariant::HistoryRecurrence
                     : rund::kernel::LoweringArtifactVariant::Recurrence;
  const std::span<const std::uint64_t> history_pitches =
      recurrence.history == nullptr ? std::span<const std::uint64_t>{}
                                    : recurrence.history->pitches();
  rund::kernel::LoweringArtifact artifact{};
  if (!recurrence.ready() || recurrence.canonical_artifact == nullptr ||
      !recurrence.source_plan.ok ||
      !rund::node::accel::detail::MaterializeMapRecurrenceArtifact(
          *recurrence.canonical_artifact, recurrence.source_plan, 2u, 1u,
          history_pitches, artifact) ||
      recurrence.iterations != 2u ||
      recurrence.plan.op_hash_hi != artifact.key.op_hash_hi ||
      recurrence.bindings.op_hash_hi != artifact.key.op_hash_hi ||
      artifact.key.variant != expected_variant ||
      recurrence.writes_each_iteration() != writes_history) {
    return false;
  }
  const bool wide = scalar == ComputeScalar::Lane64;
  const std::string lane = wide ? "64" : "32";
  constexpr std::string_view state = "read_7374617465";
  constexpr std::string_view constant = "read_636f6e7374616e74";
  const std::string invariant_address =
      "RundBase_" + std::string{constant} +
      (uniform_invariant ? std::string{}
                         : " + gid * RundStride_" + std::string{constant});
  const std::string state_load =
      api == ComputeApi::Metal
          ? "LoadI" + lane + "(" + std::string{state} + ", RundBase_" +
                std::string{state} + " + gid * RundStride_" +
                std::string{state} + ")"
          : "LoadI" + lane + "_" + std::string{state} + "(RundBase_" +
                std::string{state} + " + gid * RundStride_" +
                std::string{state} + ")";
  const std::string invariant_load =
      api == ComputeApi::Metal ? "LoadI" + lane + "(" + std::string{constant} +
                                     ", " + invariant_address + ")"
                               : "LoadI" + lane + "_" + std::string{constant} +
                                     "(" + invariant_address + ")";
  const std::string exact_boundary =
      api == ComputeApi::Metal ? (wide ? "rund_next_0 = long(node_3.lo);"
                                       : "rund_next_0 = int(node_3.lo);")
                               : (wide ? "rund_next_0 = node_3.lo;"
                                       : "rund_next_0 = uint(node_3.lo);");
  const std::string &source = artifact.source_text;
  const std::uint64_t bytes = wide ? 8u : 4u;
  std::uint64_t specialized_upper = 0u;
  constexpr std::uint64_t DecimalLiteralGrowthPerBinding = 38u;
  constexpr std::uint64_t BindingCount = 3u;
  if (artifact.source_text_upper_bytes != source.size() + 37u ||
      recurrence.source_plan.exact_source_bytes != source.size() ||
      recurrence.source_plan.source_upper_bytes !=
          artifact.source_text_upper_bytes ||
      !rund::node::accel::detail::MapSpecializedSourceUpperBytes(
          artifact, recurrence.plan, specialized_upper) ||
      specialized_upper != artifact.source_text_upper_bytes +
                               BindingCount * DecimalLiteralGrowthPerBinding ||
      artifact.retained_dynamic_memory_bytes() <
          rund::kernel::compute_retained_detail::StringExternalStorageBytes(
              source)) {
    return false;
  }
  constexpr std::string_view result = "write_726573756c74";
  const std::string history_address =
      "RundBase_" + std::string{result} + " + rund_iteration * " +
      std::to_string(bytes * 4u) + "u + gid * RundStride_" +
      std::string{result};
  const bool history_matches =
      writes_history
          ? recurrence.history != nullptr && recurrence.history->count == 1u &&
                recurrence.history->outputs[0].count == 8u &&
                recurrence.history->pitch_bytes[0] == bytes * 4u &&
                source.find(history_address) != std::string::npos &&
                source.find(", rund_carry_0);") == std::string::npos
          : recurrence.history == nullptr &&
                source.find(", rund_carry_0);") != std::string::npos;
  return history_matches && Count(source, state_load) == 1u &&
         Count(source, invariant_load) == 1u &&
         source.find("RundWideFrom" + lane + "(rund_carry_0)") !=
             std::string::npos &&
         source.find("RundWideFrom" + lane + "(rund_invariant_1)") !=
             std::string::npos &&
         source.find(exact_boundary) != std::string::npos &&
         source.find("rund_carry_0 = rund_next_0;") != std::string::npos &&
         source.find(api == ComputeApi::Metal
                         ? "rund_iteration < rund_iterations"
                         : "rund_iteration < rund_dispatch.iterations") !=
             std::string::npos;
}

[[nodiscard]] bool TransformFailureIsTransactional() {
  Fixture fixture{ComputeApi::Metal, ComputeScalar::Lane32};
  auto &artifact = fixture.occurrences.front().step.artifact;
  artifact.source_text += "// artifact_variant=canonical\n";
  artifact.source_text_upper_bytes = artifact.source_text.size() + 19u;
  const auto key_before = artifact.key;
  const std::string source_before = artifact.source_text;
  const std::uint64_t upper_before = artifact.source_text_upper_bytes;
  const std::uint64_t retained_before =
      artifact.retained_dynamic_memory_bytes();
  const std::uint64_t map_hi_before = artifact.metadata.map.op_hash_hi;
  const std::uint64_t map_lo_before = artifact.metadata.map.op_hash_lo;
  return !rund::node::accel::detail::TransformSource(artifact, 2u, 1u) &&
         artifact.key == key_before && artifact.source_text == source_before &&
         artifact.source_text_upper_bytes == upper_before &&
         artifact.retained_dynamic_memory_bytes() == retained_before &&
         artifact.metadata.map.op_hash_hi == map_hi_before &&
         artifact.metadata.map.op_hash_lo == map_lo_before;
}

} // namespace

bool MapRecurrenceSourceMaterializationContract() {
  return TransformFailureIsTransactional() &&
         SourceMatches(ComputeApi::Metal, ComputeScalar::Lane32) &&
         SourceMatches(ComputeApi::Metal, ComputeScalar::Lane64) &&
         SourceMatches(ComputeApi::Vulkan, ComputeScalar::Lane32) &&
         SourceMatches(ComputeApi::Vulkan, ComputeScalar::Lane64) &&
         SourceMatches(ComputeApi::Metal, ComputeScalar::Lane32, true) &&
         SourceMatches(ComputeApi::Metal, ComputeScalar::Lane64, true) &&
         SourceMatches(ComputeApi::Vulkan, ComputeScalar::Lane32, true) &&
         SourceMatches(ComputeApi::Vulkan, ComputeScalar::Lane64, true) &&
         SourceMatches(ComputeApi::Metal, ComputeScalar::Lane32, false, true) &&
         SourceMatches(ComputeApi::Metal, ComputeScalar::Lane64, false, true) &&
         SourceMatches(ComputeApi::Vulkan, ComputeScalar::Lane32, false,
                       true) &&
         SourceMatches(ComputeApi::Vulkan, ComputeScalar::Lane64, false, true);
}

} // namespace node_accel_contract
