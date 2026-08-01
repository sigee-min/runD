#include "src/accel/kernel/recurrence.hpp"
#include "src/accel/context/internal/execution.hpp"

#include <kernel/program/compute/lowering/text.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace node_accel_contract {
namespace {

using rund::kernel::ComputeApi;
using rund::kernel::ComputeBindingAccess;
using rund::kernel::ComputeDomain;
using rund::kernel::ComputeScalar;
using rund::kernel::LoweringArtifactKind;
using rund::node::accel::detail::BackendBatchEntry;
using rund::node::accel::detail::BackendRecurrence;
using rund::node::accel::detail::BackendRun;
using rund::node::accel::detail::BackendWindow;
using rund::node::accel::detail::BackendWindowPhase;
using rund::node::accel::detail::BoundStep;
using rund::node::accel::detail::BuildMapRecurrence;
using rund::node::accel::detail::BuildNestedMapRecurrence;
using rund::node::accel::detail::KernelExecutionStep;
using rund::node::accel::detail::MapRecurrence;
using rund::node::accel::detail::MapRecurrenceState;
using rund::node::accel::detail::PlannedStep;
using rund::node::accel::detail::RunBinds;
using rund::node::accel::detail::StepBinds;

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

[[nodiscard]] std::string Digits(const std::uint64_t value) {
  std::string out;
  rund::kernel::compute_lowering_detail::AppendHex64Digits(out, value);
  return out;
}

[[nodiscard]] std::string KeyLine(const char *const name,
                                  const std::uint64_t value) {
  std::string out = "// ";
  out += name;
  out += "=0x";
  out += Digits(value);
  out += "\n";
  return out;
}

[[nodiscard]] std::string Source(const ComputeApi api,
                                 const ComputeScalar scalar,
                                 const std::uint64_t hash,
                                 const bool uniform_invariant) {
  const bool wide = scalar == ComputeScalar::Lane64;
  const std::string lane = wide ? "64" : "32";
  constexpr std::string_view state = "read_7374617465";
  constexpr std::string_view constant = "read_636f6e7374616e74";
  constexpr std::string_view result = "write_726573756c74";
  const std::string invariant_address =
      "RundBase_" + std::string{constant} +
      (uniform_invariant
           ? std::string{}
           : " + gid * RundStride_" + std::string{constant});
  std::string source;
  source += "// artifact_variant=canonical\n";
  source += KeyLine("op_hash_hi", hash);
  source += KeyLine("op_hash_lo", hash + 1u);
  source += KeyLine("canonical_ir_hash_hi", hash);
  source += KeyLine("canonical_ir_hash_lo", hash + 1u);
  if (api == ComputeApi::Metal) {
    source += "kernel void rund_compute_map_" + Digits(hash) + "_" +
              Digits(hash + 1u) + "(\n";
    source += "    constant uchar* rund_params [[buffer(0)]],\n";
    source +=
        "    const device uchar* " + std::string{state} + " [[buffer(1)]],\n";
    source += "    const device uchar* " + std::string{constant} +
              " [[buffer(2)]],\n";
    source += "    device uchar* " + std::string{result} + " [[buffer(3)]],\n";
    source += "    uint gid [[thread_position_in_grid]]) {\n";
    source += "  const RundWide node_1 = RundWideFrom" + lane + "(LoadI" +
              lane + "(" + std::string{state} + ", RundBase_" +
              std::string{state} + " + gid * RundStride_" + std::string{state} +
              "));\n";
    source += "  const RundWide node_2 = RundWideFrom" + lane + "(LoadI" +
              lane + "(" + std::string{constant} + ", " +
              invariant_address + "));\n";
    source += "  const RundWide node_3 = RundWideAdd(node_1, node_2);\n";
    source += "  StoreI" + lane + "(" + std::string{result} + ", RundBase_" +
              std::string{result} + " + gid * RundStride_" +
              std::string{result} + ", ";
    source += wide ? "long(node_3.lo));\n" : "int(node_3.lo));\n";
  } else {
    source += "layout(push_constant) uniform RundDispatch {\n";
    source += "  uint tile_count;\n";
    source += "  uint iterations;\n";
    source += "} rund_dispatch;\n";
    source += "void main() {\n";
    source += "  const uint gid = gl_GlobalInvocationID.x;\n";
    source += "  if (gid >= rund_dispatch.tile_count) { return; }\n";
    source += "  const RundWide node_1 = RundWideFrom" + lane + "(LoadI" +
              lane + "_" + std::string{state} + "(RundBase_" +
              std::string{state} + " + gid * RundStride_" + std::string{state} +
              "));\n";
    source += "  const RundWide node_2 = RundWideFrom" + lane + "(LoadI" +
              lane + "_" + std::string{constant} + "(" +
              invariant_address + "));\n";
    source += "  const RundWide node_3 = RundWideAdd(node_1, node_2);\n";
    source += "  StoreI" + lane + "_" + std::string{result} + "(RundBase_" +
              std::string{result} + " + gid * RundStride_" +
              std::string{result} + ", ";
    source += wide ? "node_3.lo);\n" : "uint(node_3.lo));\n";
  }
  source += "}\n";
  return source;
}

struct Occurrence final {
  KernelExecutionStep step{};
  PlannedStep planned{};
  RunBinds refs{};
  StepBinds binds{};
  BoundStep bound{};
  BackendRun run{};
  std::shared_ptr<void> prepared{std::make_shared<int>(1)};
  BackendBatchEntry entry{};
};

struct Fixture final {
  std::array<std::shared_ptr<void>, 4u> owners{
      std::make_shared<int>(1), std::make_shared<int>(2),
      std::make_shared<int>(3), std::make_shared<int>(4)};
  std::array<Occurrence, 2u> occurrences{};
  std::array<BackendBatchEntry, 2u> entries{};
  std::array<std::uint8_t, 2u> barriers{0u, 1u};

  Fixture(const ComputeApi api, const ComputeScalar scalar,
          const bool uniform_invariant = false,
          const bool writes_history = false) {
    const std::uint64_t bytes =
        scalar == ComputeScalar::Lane64 ? std::uint64_t{8u} : std::uint64_t{4u};
    constexpr std::uint64_t hash = 0x1020304050607080ull;
    for (std::size_t iteration = 0u; iteration < occurrences.size();
         ++iteration) {
      Occurrence &item = occurrences[iteration];
      auto &artifact = item.step.artifact;
      artifact.key = rund::kernel::ArtifactKey{
          .api = api,
          .scalar = scalar,
          .domain = ComputeDomain::Fixed,
          .op_hash_hi = hash,
          .op_hash_lo = hash + 1u,
          .canonical_ir_hash_hi = hash,
          .canonical_ir_hash_lo = hash + 1u,
      };
      artifact.kind = api == ComputeApi::Metal
                          ? LoweringArtifactKind::MetalSource
                          : LoweringArtifactKind::VulkanSource;
      artifact.metadata.map = rund::kernel::ComputeMap{
          .op_hash_hi = hash,
          .op_hash_lo = hash + 1u,
          .api = api,
          .scalar = scalar,
          .domain = ComputeDomain::Fixed,
          .input_buffer_count = 2u,
          .output_buffer_count = 1u,
          .input_bytes_per_tile = bytes * 2u,
          .output_bytes_per_tile = bytes,
      };
      artifact.metadata.input_element_bytes = {bytes, bytes};
      artifact.metadata.output_element_bytes = {bytes};
      artifact.metadata.binding_accesses = {ComputeBindingAccess::Read,
                                            ComputeBindingAccess::Read,
                                            ComputeBindingAccess::Write};
      artifact.metadata.binding_names = {"state", "constant", "result"};
      artifact.metadata.read_count = 2u;
      artifact.metadata.direct_read_mask = uniform_invariant ? 0x1u : 0x3u;
      artifact.metadata.uniform_read_mask = uniform_invariant ? 0x2u : 0u;
      artifact.metadata.write_count = 1u;
      artifact.metadata.ok = true;
      artifact.metadata.reason = "ok";
      artifact.source_text = Source(api, scalar, hash, uniform_invariant);
      artifact.ok = true;
      artifact.reason = "ok";
      item.step.map_semantic = rund::node::accel::detail::MapSemantic{
          .kind = rund::node::accel::detail::MapSemanticKind::AddWrapU32Pair,
          .recurrence_total = true,
      };

      item.planned.plan = rund::kernel::ComputePlan{
          .tile_count = 4u,
          .op_hash_hi = hash,
          .op_hash_lo = hash + 1u,
          .api = api,
          .scalar = scalar,
          .domain = ComputeDomain::Fixed,
          .input_buffer_count = 2u,
          .output_buffer_count = 1u,
          .input_bytes_per_tile = bytes * 2u,
          .output_bytes_per_tile = bytes,
          .bytes_per_tile = bytes * 3u,
          .staging_bytes = bytes * 12u,
          .dispatch_window_tiles = 4u,
          .dispatch_count = 1u,
          .ok = true,
          .reason = "ok",
      };
      item.planned.artifact = &item.step.artifact;

      const std::uint64_t input_owner = iteration == 0u ? 0u : 1u;
      const std::uint64_t output_owner =
          writes_history ? 1u : (iteration == 0u ? 1u : 2u);
      const auto ref = [&](const std::uint64_t owner,
                           const std::uint32_t usage) {
        return rund::kernel::ResidentBufferRef{
            .id = owner + 1u,
            .bytes = writes_history && owner == 1u ? bytes * 8u : bytes * 4u,
            .element_bytes = bytes,
            .stride_bytes = bytes,
            .count = 4u,
            .usage = usage,
        };
      };
      item.refs.reserve(3u);
      (void)item.refs.push(ref(input_owner, rund::kernel::kResidentUsageRead),
                           owners[input_owner]);
      auto invariant = ref(3u, rund::kernel::kResidentUsageRead);
      if (uniform_invariant) {
        invariant.bytes = bytes;
        invariant.count = 1u;
      }
      (void)item.refs.push(invariant, owners[3]);
      auto output = ref(output_owner, rund::kernel::kResidentUsageWrite);
      if (writes_history) {
        output.offset_bytes = iteration * bytes * 4u;
      }
      (void)item.refs.push(output, owners[output_owner]);
      item.binds.inputs.bind(item.refs, 2u);
      item.binds.outputs.bind(item.refs, 1u);
      (void)item.binds.inputs.push(0u);
      (void)item.binds.inputs.push(1u);
      (void)item.binds.outputs.push(2u);
      item.bound = BoundStep{
          .step = &item.step,
          .planned = &item.planned,
          .bindings = item.binds,
      };
      item.bound.map_windows.inline_windows[0] =
          rund::kernel::ComputeDispatchWindow{.tile_count = 4u};
      item.bound.map_windows.window_count = 1u;
      item.run.steps = &item.bound;
      item.run.step_count = 1u;
      item.entry = BackendBatchEntry{
          .run = &item.run,
          .prepared = &item.prepared,
          .recurrence =
              BackendRecurrence{
                  .logical_step = 7u,
                  .iteration = static_cast<std::uint32_t>(iteration),
                  .bound = 2u,
                  .writes_each_iteration = writes_history,
              },
      };
      entries[iteration] = item.entry;
    }
  }
};

[[nodiscard]] bool SourceMatches(const ComputeApi api,
                                 const ComputeScalar scalar,
                                 const bool uniform_invariant = false,
                                 const bool writes_history = false) {
  Fixture fixture{api, scalar, uniform_invariant, writes_history};
  const MapRecurrence recurrence =
      BuildMapRecurrence(fixture.entries, fixture.barriers);
  const auto expected_variant =
      writes_history
          ? rund::kernel::LoweringArtifactVariant::HistoryRecurrence
          : rund::kernel::LoweringArtifactVariant::Recurrence;
  if (!recurrence.ready() || recurrence.iterations != 2u ||
      recurrence.plan.op_hash_hi != recurrence.artifact.key.op_hash_hi ||
      recurrence.bindings.op_hash_hi != recurrence.artifact.key.op_hash_hi ||
      recurrence.artifact.key.variant != expected_variant ||
      recurrence.writes_each_iteration() != writes_history) {
    return false;
  }
  const bool wide = scalar == ComputeScalar::Lane64;
  const std::string lane = wide ? "64" : "32";
  constexpr std::string_view state = "read_7374617465";
  constexpr std::string_view constant = "read_636f6e7374616e74";
  const std::string invariant_address =
      "RundBase_" + std::string{constant} +
      (uniform_invariant
           ? std::string{}
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
      api == ComputeApi::Metal
          ? "LoadI" + lane + "(" + std::string{constant} + ", " +
                invariant_address + ")"
          : "LoadI" + lane + "_" + std::string{constant} + "(" +
                invariant_address + ")";
  const std::string exact_boundary =
      api == ComputeApi::Metal ? (wide ? "rund_next_0 = long(node_3.lo);"
                                       : "rund_next_0 = int(node_3.lo);")
                               : (wide ? "rund_next_0 = node_3.lo;"
                                       : "rund_next_0 = uint(node_3.lo);");
  const std::string &source = recurrence.artifact.source_text;
  const std::uint64_t bytes = wide ? 8u : 4u;
  constexpr std::string_view result = "write_726573756c74";
  const std::string history_address =
      "RundBase_" + std::string{result} + " + rund_iteration * " +
      std::to_string(bytes * 4u) + "u + gid * RundStride_" +
      std::string{result};
  const bool history_matches =
      writes_history
          ? recurrence.history != nullptr &&
                recurrence.history->outputs.size() == 1u &&
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

[[nodiscard]] bool HistoryMarkerContract() {
  Fixture ready{ComputeApi::Metal, ComputeScalar::Lane32, false, true};
  const MapRecurrence history =
      BuildMapRecurrence(ready.entries, ready.barriers);
  if (!history.ready() || !history.writes_each_iteration()) {
    return false;
  }

  Fixture bad_pitch{ComputeApi::Metal, ComputeScalar::Lane32, false, true};
  bad_pitch.occurrences[1u].refs.inline_refs[2u].offset_bytes += 4u;
  if (BuildMapRecurrence(bad_pitch.entries, bad_pitch.barriers).state !=
      MapRecurrenceState::Invalid) {
    return false;
  }

  Fixture non_total{ComputeApi::Metal, ComputeScalar::Lane32, false, true};
  for (Occurrence &occurrence : non_total.occurrences) {
    occurrence.step.map_semantic.recurrence_total = false;
  }
  if (BuildMapRecurrence(non_total.entries, non_total.barriers).state !=
      MapRecurrenceState::Ineligible) {
    return false;
  }

  Fixture mixed_marker{ComputeApi::Metal, ComputeScalar::Lane32, false, true};
  mixed_marker.entries[1u].recurrence.writes_each_iteration = false;
  return BuildMapRecurrence(mixed_marker.entries, mixed_marker.barriers).state ==
         MapRecurrenceState::Ineligible;
}

[[nodiscard]] bool NestedMarkerContract() {
  Fixture fixture{ComputeApi::Metal, ComputeScalar::Lane32};
  std::array<BackendWindow, 2u> windows{};
  for (std::size_t index = 0u; index < windows.size(); ++index) {
    windows[index] = BackendWindow{
        .maximum = 8u,
        .tile = 4u,
        .iteration = 1u,
        .bound = 2u,
        .state = 3u,
        .outer_iteration = 1u,
        .outer_bound = 2u,
        .inner_iteration = static_cast<std::uint32_t>(index),
        .inner_bound = 2u,
        .inner_advance = 1u,
        .phase = BackendWindowPhase::NestedAction,
    };
    fixture.entries[index].recurrence.window = &windows[index];
  }
  fixture.barriers = {1u, 1u};
  const MapRecurrence ready =
      BuildNestedMapRecurrence(fixture.entries, fixture.barriers);
  if (!ready.ready() || ready.iterations != windows.size()) {
    return false;
  }

  fixture.barriers[1u] = 0u;
  if (BuildNestedMapRecurrence(fixture.entries, fixture.barriers).state !=
      MapRecurrenceState::Invalid) {
    return false;
  }
  fixture.barriers[1u] = 1u;
  windows[1u].inner_iteration = 0u;
  if (BuildNestedMapRecurrence(fixture.entries, fixture.barriers).state !=
      MapRecurrenceState::Ineligible) {
    return false;
  }
  windows[1u].inner_iteration = 1u;
  fixture.occurrences[0u].step.artifact.metadata.read_routes.push_back(
      rund::kernel::ReadRoute{.source = 0u, .index = 1u, .count = 4u});
  if (BuildNestedMapRecurrence(fixture.entries, fixture.barriers).state !=
      MapRecurrenceState::Ineligible) {
    return false;
  }
  return true;
}

} // namespace

[[nodiscard]] bool MapRecurrenceSourceContract() {
  Fixture ordinary{ComputeApi::Metal, ComputeScalar::Lane32};
  ordinary.entries[0].recurrence.bound = 1u;
  ordinary.entries[1].recurrence.bound = 1u;
  if (BuildMapRecurrence(ordinary.entries, ordinary.barriers).state !=
      rund::node::accel::detail::MapRecurrenceState::Ineligible) {
    return false;
  }
  return NestedMarkerContract() && HistoryMarkerContract() &&
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
