#include "contract/program/compute/local.hpp"
#include "contract/support/allocation.hpp"
#include "test/assert.hpp"

#include <kernel/program/compute/lowering/emission.hpp>
#include <kernel/program/compute/plan.hpp>

#include <array>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace program_compute_contract {

namespace {

[[nodiscard]] auto BuildI32AdmissionOp() {
  rund::kernel::i32 input[4]{};
  rund::kernel::i32 output[4]{};
  const auto body =
      rund::compute_dsl::bind(4u).i32().read<"input">(input).write<"output">(
          output);
  return rund::compute_dsl::def("admission-i32")
      .on(body)
      .map([](auto index, auto bindings) {
        auto input = bindings.template read<"input">();
        auto output = bindings.template write<"output">();
        output[index] = input[index] + rund::kernel::i32{1};
      });
}

[[nodiscard]] auto BuildU64AdmissionOp() {
  rund::kernel::u64 input[4]{};
  rund::kernel::u64 output[4]{};
  const auto body =
      rund::compute_dsl::bind(4u).u64().read<"input">(input).write<"output">(
          output);
  return rund::compute_dsl::def("admission-u64")
      .on(body)
      .map([](auto index, auto bindings) {
        auto input = bindings.template read<"input">();
        auto output = bindings.template write<"output">();
        output[index] = input[index] + rund::kernel::u64{1u};
      });
}

template <typename Op>
[[nodiscard]] rund::kernel::ComputePlan
AdmissionPlan(const Op &op, const rund::kernel::ComputeApi api) {
  rund::kernel::ComputeMap map = op.map();
  map.api = api;
  return rund::kernel::PlanCompute(
      rund::kernel::TilePhaseDescription{
          .phase_id = 71u,
          .tile_count = 4u,
          .capacity =
              rund::kernel::TilePhaseCapacityRequirement{
                  .output_shards = 4u,
                  .queue_slots = 4u,
                  .task_slots = 4u,
              },
      },
      map,
      rund::kernel::ComputeCaps{
          .api = api,
          .device_bytes = 1024u * 1024u,
          .staging_bytes = 1024u * 1024u,
          .max_window_tiles = 4u,
          .subgroup_width = 32u,
          .ok = true,
          .reason = "ok",
      },
      rund::kernel::ComputeLimit{
          .staging_bytes = 1024u * 1024u,
          .max_window_tiles = 4u,
      });
}

template <typename Op> void CheckArtifactAdmissionParity(const Op &op) {
  using rund::kernel::compute_lowering_detail::AdmitArtifact;
  constexpr std::array apis{
      rund::kernel::ComputeApi::Cpu,
      rund::kernel::ComputeApi::Metal,
      rund::kernel::ComputeApi::Vulkan,
  };
  for (const rund::kernel::ComputeApi api : apis) {
    const rund::kernel::ComputePlan plan = AdmissionPlan(op, api);
    auto retained =
        rund::kernel::compute_lowering_detail::LowerRetainedComputeArtifact(
            op.ir(), api);
    const rund::kernel::LoweringArtifact artifact =
        rund::kernel::LowerComputeIR(op.ir(), api);
    TEST_ASSERT(plan.ok);
    TEST_ASSERT(artifact.ok);
    TEST_ASSERT(retained.artifact.ok);
    TEST_ASSERT(retained.input.ok);
    TEST_ASSERT(retained.parse_count() == 1u);
    TEST_ASSERT(retained.emission_count == 1u);
    TEST_ASSERT(retained.artifact.canonical_ir_bytes.empty());

    const rund::kernel::ComputeIR planned =
        rund::kernel::compute_lowering_detail::PlanIR(plan);
    TEST_ASSERT(planned.scalar == plan.scalar);
    TEST_ASSERT(planned.domain == plan.domain);
    TEST_ASSERT(planned.fixed_format == plan.fixed_format);
    TEST_ASSERT(planned.op_hash_hi == plan.op_hash_hi);
    TEST_ASSERT(planned.op_hash_lo == plan.op_hash_lo);
    TEST_ASSERT(planned.canonical_bytes.empty());
    TEST_ASSERT(planned.ok);

    const auto parsed_bytes = [](const auto &input) {
      using rund::kernel::compute_retained_detail::Add;
      using rund::kernel::compute_retained_detail::StringExternalStorageBytes;
      using rund::kernel::compute_retained_detail::VectorCapacityBytes;
      rund::kernel::u64 bytes = StringExternalStorageBytes(input.parsed.name);
      bytes = Add(bytes, VectorCapacityBytes(input.parsed.bindings));
      for (const auto &binding : input.parsed.bindings) {
        bytes = Add(bytes, StringExternalStorageBytes(binding.name));
        bytes = Add(bytes, VectorCapacityBytes(binding.value_bytes));
      }
      return Add(bytes, VectorCapacityBytes(input.parsed.nodes));
    };
    TEST_ASSERT(retained.input.retained_dynamic_memory_bytes() ==
                parsed_bytes(retained.input));

    const rund::kernel::compute_lowering_detail::ComputeInputAdmission
        *cpu_input = nullptr;
    if (api == rund::kernel::ComputeApi::Cpu) {
      std::string{}.swap(retained.artifact.source_text);
      cpu_input = &retained.input;
    }
    const auto warm =
        rund::kernel::compute_lowering_detail::AdmitRetained(
            plan, retained.artifact, cpu_input);
    TEST_ASSERT(warm.ok);
    TEST_ASSERT(warm.parse_count == 0u);
    TEST_ASSERT(warm.emission_count == 0u);

    rund::kernel::LoweringArtifact retained_key_forged = retained.artifact;
    ++retained_key_forged.key.op_hash_lo;
    const auto retained_key_rejected =
        rund::kernel::compute_lowering_detail::AdmitRetained(
            plan, retained_key_forged, cpu_input);
    TEST_ASSERT(!retained_key_rejected.ok);
    TEST_ASSERT(retained_key_rejected.parse_count == 0u);
    TEST_ASSERT(retained_key_rejected.emission_count == 0u);

    rund::kernel::LoweringArtifact retained_metadata_forged = retained.artifact;
    ++retained_metadata_forged.metadata.map.input_buffer_count;
    const auto retained_metadata_rejected =
        rund::kernel::compute_lowering_detail::AdmitRetained(
            plan, retained_metadata_forged, cpu_input);
    TEST_ASSERT(!retained_metadata_rejected.ok);
    TEST_ASSERT(retained_metadata_rejected.parse_count == 0u);
    TEST_ASSERT(retained_metadata_rejected.emission_count == 0u);

    if (api == rund::kernel::ComputeApi::Cpu) {
      auto retained_input_forged = retained.input;
      ++retained_input_forged.key.op_hash_lo;
      const auto retained_input_rejected =
          rund::kernel::compute_lowering_detail::AdmitRetained(
              plan, retained.artifact, &retained_input_forged);
      TEST_ASSERT(!retained_input_rejected.ok);
      TEST_ASSERT(retained_input_rejected.parse_count == 0u);
      TEST_ASSERT(retained_input_rejected.emission_count == 0u);
    } else {
      const auto retained_input_rejected =
          rund::kernel::compute_lowering_detail::AdmitRetained(
              plan, retained.artifact, &retained.input);
      TEST_ASSERT(!retained_input_rejected.ok);
      TEST_ASSERT(retained_input_rejected.parse_count == 0u);
      TEST_ASSERT(retained_input_rejected.emission_count == 0u);
    }

    const auto admitted = AdmitArtifact(plan, artifact);
    TEST_ASSERT(admitted.ok);
    TEST_ASSERT(admitted.parse_count() == 1u);
    TEST_ASSERT(admitted.emission_count == 1u);

    rund::kernel::ComputePlan binding_count_forged = plan;
    ++binding_count_forged.input_buffer_count;
    const auto binding_count_rejected =
        AdmitArtifact(binding_count_forged, artifact);
    TEST_ASSERT(!binding_count_rejected.ok);
    TEST_ASSERT(binding_count_rejected.parse_count() == 1u);
    TEST_ASSERT(binding_count_rejected.emission_count == 1u);

    rund::kernel::LoweringArtifact key_forged = artifact;
    ++key_forged.key.op_hash_lo;
    const auto key_rejected = AdmitArtifact(plan, key_forged);
    TEST_ASSERT(!key_rejected.ok);
    TEST_ASSERT(key_rejected.parse_count() == 0u);
    TEST_ASSERT(key_rejected.emission_count == 0u);

    rund::kernel::LoweringArtifact canonical_forged = artifact;
    canonical_forged.canonical_ir_bytes.push_back(0xffu);
    const auto canonical_rejected =
        AdmitArtifact(plan, canonical_forged);
    TEST_ASSERT(!canonical_rejected.ok);
    TEST_ASSERT(canonical_rejected.parse_count() == 0u);
    TEST_ASSERT(canonical_rejected.emission_count == 0u);

    rund::kernel::LoweringArtifact source_forged = artifact;
    source_forged.source_text += "\nforged";
    const auto source_rejected = AdmitArtifact(plan, source_forged);
    TEST_ASSERT(!source_rejected.ok);
    TEST_ASSERT(source_rejected.parse_count() == 1u);
    TEST_ASSERT(source_rejected.emission_count == 1u);

    rund::kernel::LoweringArtifact metadata_forged = artifact;
    TEST_ASSERT(!metadata_forged.metadata.binding_names.empty());
    metadata_forged.metadata.binding_names.front() += "-forged";
    const auto metadata_rejected = AdmitArtifact(plan, metadata_forged);
    TEST_ASSERT(!metadata_rejected.ok);
    TEST_ASSERT(metadata_rejected.parse_count() == 1u);
    TEST_ASSERT(metadata_rejected.emission_count == 1u);

    rund::kernel::LoweringArtifact null_reason_forged = artifact;
    null_reason_forged.metadata.reason = nullptr;
    const auto null_reason_rejected =
        AdmitArtifact(plan, null_reason_forged);
    TEST_ASSERT(!null_reason_rejected.ok);
    TEST_ASSERT(null_reason_rejected.parse_count() == 1u);
    TEST_ASSERT(null_reason_rejected.emission_count == 1u);
  }
}

void CheckParseCapacityBoundary(const rund::compute_dsl::ComputeOp &op) {
  using rund::kernel::compute_lowering_detail::GuardComputeIRParse;
  using rund::kernel::compute_lowering_detail::ParseComputeIR;
  using rund::kernel::compute_lowering_detail::Reader;

  constexpr std::size_t string_size = 64u;
  std::vector<rund::kernel::u8> encoded(4u + string_size,
                                        static_cast<rund::kernel::u8>('x'));
  encoded[0] = static_cast<rund::kernel::u8>(string_size);
  encoded[1] = 0u;
  encoded[2] = 0u;
  encoded[3] = 0u;
  Reader reader{encoded};
  std::string decoded{};
  kernel_contract_test::memory_allocation::Reset();
  const bool decoded_ok = reader.read_string(decoded);
  kernel_contract_test::memory_allocation::Stop();
  TEST_ASSERT(decoded_ok);
  TEST_ASSERT(decoded == std::string(string_size, 'x'));
  TEST_ASSERT(kernel_contract_test::memory_allocation::Count() == 1u);

  const auto length_rejected = GuardComputeIRParse(
      []() -> rund::kernel::compute_lowering_detail::ParsedIR {
        throw std::length_error{"contract"};
      });
  TEST_ASSERT(!length_rejected.ok);
  TEST_ASSERT(std::string_view{length_rejected.reason} ==
              "compute_ir_capacity");

  kernel_contract_test::memory_allocation::FailNext();
  const auto allocation_rejected = ParseComputeIR(op.ir());
  TEST_ASSERT(!allocation_rejected.ok);
  TEST_ASSERT(std::string_view{allocation_rejected.reason} ==
              "compute_ir_capacity");
}

} // namespace

int RunComputeLoweringContract() {
  const auto op = lowering_support::BuildFixedLane32Op(7);
  CheckParseCapacityBoundary(op);
  const rund::kernel::compute_lowering_detail::ComputeInputAdmission input =
      rund::kernel::compute_lowering_detail::AdmitComputeInput(
          op.ir(), rund::kernel::ComputeApi::Cpu);
  TEST_ASSERT(input.ok);
  TEST_ASSERT(input.parse_count == 1u);
  const auto direct =
      rund::kernel::compute_lowering_detail::EmitComputeArtifactTransient(
          op.ir(), input);
  TEST_ASSERT(direct.ok);
  TEST_ASSERT(direct.emission_count == 1u);
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);

  TEST_ASSERT(artifact.ok);
  TEST_ASSERT(artifact.key == direct.key);
  TEST_ASSERT(artifact.kind == direct.kind);
  TEST_ASSERT(artifact.source_text == direct.source_text);
  TEST_ASSERT(artifact.kind == rund::kernel::LoweringArtifactKind::CpuPlan);
  TEST_ASSERT(artifact.key.api == rund::kernel::ComputeApi::Cpu);
  TEST_ASSERT(artifact.key.scalar == rund::kernel::ComputeScalar::Lane32);
  TEST_ASSERT(artifact.key.op_hash_hi == op.ir().op_hash_hi);
  TEST_ASSERT(artifact.key.op_hash_lo == op.ir().op_hash_lo);
  TEST_ASSERT(artifact.canonical_ir_bytes == op.ir().canonical_bytes);
  TEST_ASSERT(artifact.metadata.ok);
  TEST_ASSERT(artifact.metadata.map.op_hash_hi ==
              direct.metadata.map.op_hash_hi);
  TEST_ASSERT(artifact.metadata.map.op_hash_lo ==
              direct.metadata.map.op_hash_lo);
  TEST_ASSERT(artifact.source_text.find("rund.compute.cpu.plan") !=
              std::string_view::npos);
  TEST_ASSERT(artifact.source_text.find("api=cpu") != std::string_view::npos);
  TEST_ASSERT(artifact.source_text.find("binding[0].kind=param") !=
              std::string_view::npos);
  TEST_ASSERT(artifact.source_text.find("node[") != std::string_view::npos);
  TEST_ASSERT(artifact.source_text.find("].op=mul") != std::string_view::npos);

  CheckArtifactAdmissionParity(lowering_support::BuildFixedLane32Op(7));
  CheckArtifactAdmissionParity(lowering_support::BuildFixedLane64Op(7));
  CheckArtifactAdmissionParity(BuildI32AdmissionOp());
  CheckArtifactAdmissionParity(BuildU64AdmissionOp());
  return 0;
}

} // namespace program_compute_contract
