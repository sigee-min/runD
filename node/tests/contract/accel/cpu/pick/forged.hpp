#pragma once

#include "plan.hpp"

#include <kernel/program/compute/lowering/artifact/admission.hpp>

#include <string_view>

namespace node_accel_contract::cpu::pick {

[[nodiscard]] inline bool
RejectsForgedWithoutWriting(Work &work, const Resources &resources,
                            const rund::kernel::LoweringArtifact &forged,
                            const std::string_view reason,
                            const rund::kernel::u32 expected_parse_count,
                            const rund::kernel::u32 expected_emission_count) {
  for (rund::kernel::i32 &value : work.out) {
    value = -123;
  }
  const auto admission =
      rund::kernel::compute_lowering_detail::AdmitArtifact(
          resources.plan, forged);
  TEST_ASSERT(!admission.ok);
  TEST_ASSERT(std::string_view{admission.reason} == reason);
  TEST_ASSERT(admission.parse_count() == expected_parse_count);
  TEST_ASSERT(admission.emission_count == expected_emission_count);
  TEST_ASSERT(!resources.pick.backend.execute(
      resources.pick.backend.context, resources.plan, forged, &resources.window,
      1u, resources.bindings));
  TEST_ASSERT(resources.pick.backend.last_error != nullptr);
  TEST_ASSERT(std::string_view{resources.pick.backend.last_error(
                  resources.pick.backend.context)} == reason);
  for (const rund::kernel::i32 value : work.out) {
    TEST_ASSERT(value == -123);
  }
  return true;
}

[[nodiscard]] inline bool RejectsForgedArtifacts(Work &work,
                                                 const Resources &resources) {
  rund::kernel::LoweringArtifact key_forged = resources.artifact;
  ++key_forged.key.op_hash_lo;
  const auto key_admission =
      rund::kernel::compute_lowering_detail::AdmitArtifact(
          resources.plan, key_forged);
  TEST_ASSERT(!key_admission.ok);
  TEST_ASSERT(std::string_view{key_admission.reason} ==
              "compute_artifact_mismatch");
  TEST_ASSERT(key_admission.parse_count() == 0u);
  TEST_ASSERT(key_admission.emission_count == 0u);

  rund::kernel::LoweringArtifact source_forged = resources.artifact;
  source_forged.source_text += "\nforged-source";
  TEST_ASSERT(RejectsForgedWithoutWriting(work, resources, source_forged,
                                          "compute_artifact_mismatch", 1u, 1u));

  rund::kernel::LoweringArtifact metadata_forged = resources.artifact;
  TEST_ASSERT(!metadata_forged.metadata.binding_names.empty());
  metadata_forged.metadata.binding_names.front() += "-forged";
  TEST_ASSERT(RejectsForgedWithoutWriting(work, resources, metadata_forged,
                                          "compute_artifact_mismatch", 1u, 1u));

  rund::kernel::LoweringArtifact canonical_forged = resources.artifact;
  canonical_forged.canonical_ir_bytes.push_back(0xffu);
  TEST_ASSERT(RejectsForgedWithoutWriting(work, resources, canonical_forged,
                                          "compute_ir_hash_mismatch", 0u, 0u));
  return true;
}

} // namespace node_accel_contract::cpu::pick
