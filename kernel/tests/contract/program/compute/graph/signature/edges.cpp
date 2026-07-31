#include "test/compute/fixed.hpp"
#include "contract/program/compute/graph/local.hpp"
#include "contract/program/compute/lowering/support.hpp"
#include <kernel/program/compute/factor/identity.hpp>
#include <kernel/program/compute/factor/plan.hpp>
#include <kernel/program/compute/factor/reference.hpp>
#include <kernel/program/compute/factor/model.hpp>
#include <kernel/program/compute/graph/signature.hpp>
#include <kernel/program/compute/scatter.hpp>
#include <kernel/program/compute/solve/identity.hpp>
#include <kernel/program/compute/solve/plan.hpp>
#include <kernel/program/compute/solve/reference.hpp>
#include <kernel/program/compute/solve/model.hpp>
#include <kernel/program/compute/spectrum/identity.hpp>
#include <kernel/program/compute/spectrum/plan.hpp>
#include <kernel/program/compute/spectrum/reference.hpp>
#include <kernel/program/compute/spectrum/model.hpp>

namespace program_compute_contract {

namespace {

[[nodiscard]] bool IsEdge(const rund::kernel::GraphValueType& value,
                          const rund::kernel::GraphValueKind kind,
                          const rund::kernel::BufferRole role,
                          const rund::kernel::u32 element_bytes,
                          const rund::kernel::u64 count) noexcept {
  return value.kind == kind && value.role == role &&
         value.element_bytes == element_bytes && value.count == count;
}

[[nodiscard]] bool SameSignature(
    const rund::kernel::GraphSignature& lhs,
    const rund::kernel::GraphSignature& rhs) noexcept {
  if (lhs.ok != rhs.ok || lhs.kind != rhs.kind ||
      lhs.value_count != rhs.value_count ||
      lhs.output_count != rhs.output_count ||
      lhs.status_count != rhs.status_count) {
    return false;
  }
  for (rund::kernel::u64 index = 0u; index < lhs.value_count; ++index) {
    const rund::kernel::GraphValueType& a = lhs.values[index];
    const rund::kernel::GraphValueType& b = rhs.values[index];
    if (a.kind != b.kind || a.role != b.role ||
        a.element_bytes != b.element_bytes || a.count != b.count ||
        a.rows != b.rows || a.cols != b.cols ||
        a.batch_count != b.batch_count) {
      return false;
    }
  }
  return true;
}

int test_map_signature_builds_without_execution_metadata_payload() {
  const auto op = lowering_support::BuildFixedLane32Op(7);
  const rund::kernel::ExecutionMetadata metadata =
      rund::kernel::BuildExecutionMetadata(op.ir(),
                                           rund::kernel::ComputeApi::Metal);
  const rund::kernel::GraphSignature expected =
      rund::kernel::GraphSignatureFor(metadata);
  const rund::kernel::GraphSignature signature =
      rund::kernel::BuildMapGraphSignature(op.ir(),
                                           rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(metadata.ok);
  TEST_ASSERT(signature.ok);
  TEST_ASSERT(SameSignature(signature, expected));
  TEST_ASSERT(signature.kind == rund::kernel::NodeKind::Map);
  TEST_ASSERT(signature.value_count == 3u);
  TEST_ASSERT(IsEdge(signature.values[0u], rund::kernel::GraphValueKind::Values,
                     rund::kernel::BufferRole::Read, sizeof(rund::kernel::u32),
                     0u));
  TEST_ASSERT(IsEdge(signature.values[1u], rund::kernel::GraphValueKind::Values,
                     rund::kernel::BufferRole::Read, sizeof(rund::kernel::u32),
                     0u));
  TEST_ASSERT(IsEdge(signature.values[2u], rund::kernel::GraphValueKind::Output,
                     rund::kernel::BufferRole::Write,
                     sizeof(rund::kernel::u32), 0u));
  return 0;
}

int test_factor_signature_closes_lu_outputs() {
  const rund::kernel::FactorDesc desc{
      .op = rund::kernel::FactorOp::LU,
      .rows = 3u,
      .cols = 3u,
      .batch_count = 2u,
      .element_bytes = sizeof(rund::kernel::u32),
      .fixed_format = rund::kernel::PrimitiveFixedFormat(
          test::FixedFormatForLane(
              rund::kernel::ComputeScalar::Lane32),
          rund::kernel::ComputeApproximation::Deterministic),
  };
  const rund::kernel::FactorPlan plan = rund::kernel::PlanFactor(desc);
  const rund::kernel::GraphSignature signature =
      rund::kernel::GraphSignatureFor(plan);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(signature.ok);
  TEST_ASSERT(signature.kind == rund::kernel::NodeKind::Factor);
  TEST_ASSERT(signature.value_count == 4u);
  TEST_ASSERT(IsEdge(signature.values[0u], rund::kernel::GraphValueKind::Matrix,
                     rund::kernel::BufferRole::Read, plan.element_bytes,
                     plan.input_count));
  TEST_ASSERT(IsEdge(signature.values[1u], rund::kernel::GraphValueKind::Factor,
                     rund::kernel::BufferRole::Write, plan.element_bytes,
                     plan.factor_count));
  TEST_ASSERT(IsEdge(signature.values[2u], rund::kernel::GraphValueKind::Aux,
                     rund::kernel::BufferRole::Write,
                     sizeof(rund::kernel::u32), plan.aux_count));
  TEST_ASSERT(IsEdge(signature.values[3u], rund::kernel::GraphValueKind::Status,
                     rund::kernel::BufferRole::Write,
                     sizeof(rund::kernel::u32), plan.status_count));
  return 0;
}

int test_solve_signature_closes_factor_reuse_inputs() {
  const rund::kernel::SolveDesc desc{
      .op = rund::kernel::SolveOp::Linear,
      .input = rund::kernel::SolveInput::Factor,
      .factor = rund::kernel::FactorOp::LU,
      .rows = 4u,
      .rhs_cols = 2u,
      .batch_count = 1u,
      .element_bytes = sizeof(rund::kernel::u64),
      .fixed_format = rund::kernel::PrimitiveFixedFormat(
          test::FixedFormatForLane(
              rund::kernel::ComputeScalar::Lane64),
          rund::kernel::ComputeApproximation::Deterministic),
  };
  const rund::kernel::SolvePlan plan = rund::kernel::PlanSolve(desc);
  const rund::kernel::GraphSignature signature =
      rund::kernel::GraphSignatureFor(plan);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(signature.ok);
  TEST_ASSERT(signature.kind == rund::kernel::NodeKind::Solve);
  TEST_ASSERT(signature.value_count == 5u);
  TEST_ASSERT(IsEdge(signature.values[0u], rund::kernel::GraphValueKind::Factor,
                     rund::kernel::BufferRole::Read, plan.element_bytes,
                     plan.factor_count));
  TEST_ASSERT(IsEdge(signature.values[1u], rund::kernel::GraphValueKind::Aux,
                     rund::kernel::BufferRole::Read,
                     sizeof(rund::kernel::u32), plan.aux_count));
  TEST_ASSERT(IsEdge(signature.values[2u], rund::kernel::GraphValueKind::Rhs,
                     rund::kernel::BufferRole::Read, plan.element_bytes,
                     plan.rhs_count));
  TEST_ASSERT(IsEdge(signature.values[3u], rund::kernel::GraphValueKind::Output,
                     rund::kernel::BufferRole::Write, plan.element_bytes,
                     plan.output_count));
  TEST_ASSERT(IsEdge(signature.values[4u], rund::kernel::GraphValueKind::Status,
                     rund::kernel::BufferRole::Write,
                     sizeof(rund::kernel::u32), plan.status_count));
  return 0;
}

int test_spectrum_signature_closes_vector_output() {
  const rund::kernel::SpectrumDesc desc{
      .op = rund::kernel::SpectrumOp::SVD,
      .domain = rund::kernel::SpectrumDomain::GeneralReal,
      .vectors = rund::kernel::SpectrumVectors::Full,
      .rows = 3u,
      .cols = 2u,
      .batch_count = 2u,
      .max_iterations = 12u,
      .element_bytes = sizeof(rund::kernel::u32),
      .fixed_format = rund::kernel::PrimitiveFixedFormat(
          test::FixedFormatForLane(
              rund::kernel::ComputeScalar::Lane32),
          rund::kernel::ComputeApproximation::Deterministic),
  };
  const rund::kernel::SpectrumPlan plan = rund::kernel::PlanSpectrum(desc);
  const rund::kernel::GraphSignature signature =
      rund::kernel::GraphSignatureFor(plan);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(signature.ok);
  TEST_ASSERT(signature.kind == rund::kernel::NodeKind::Spectrum);
  TEST_ASSERT(signature.value_count == 4u);
  TEST_ASSERT(IsEdge(signature.values[0u], rund::kernel::GraphValueKind::Matrix,
                     rund::kernel::BufferRole::Read, plan.element_bytes,
                     plan.input_count));
  TEST_ASSERT(IsEdge(signature.values[1u], rund::kernel::GraphValueKind::Values,
                     rund::kernel::BufferRole::Write, plan.element_bytes,
                     plan.value_count));
  TEST_ASSERT(IsEdge(signature.values[2u], rund::kernel::GraphValueKind::Vectors,
                     rund::kernel::BufferRole::Write, plan.element_bytes,
                     plan.vector_count));
  TEST_ASSERT(IsEdge(signature.values[3u], rund::kernel::GraphValueKind::Status,
                     rund::kernel::BufferRole::Write,
                     sizeof(rund::kernel::u32), plan.status_count));
  return 0;
}

int test_scatter_signature_closes_indexed_move() {
  const rund::kernel::ScatterDesc desc{
      .element = rund::kernel::ScatterElement::U64,
      .element_count = 5u,
      .output_count = 8u,
  };
  const rund::kernel::ScatterPlan plan = rund::kernel::PlanScatter(desc);
  const rund::kernel::GraphSignature signature =
      rund::kernel::GraphSignatureFor(plan);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(signature.ok);
  TEST_ASSERT(signature.kind == rund::kernel::NodeKind::Scatter);
  TEST_ASSERT(signature.value_count == 3u);
  TEST_ASSERT(IsEdge(signature.values[0u], rund::kernel::GraphValueKind::Values,
                     rund::kernel::BufferRole::Read, plan.element_bytes,
                     plan.element_count));
  TEST_ASSERT(IsEdge(signature.values[1u],
                     rund::kernel::GraphValueKind::Indices,
                     rund::kernel::BufferRole::Read, plan.index_bytes,
                     plan.element_count));
  TEST_ASSERT(IsEdge(signature.values[2u], rund::kernel::GraphValueKind::Output,
                     rund::kernel::BufferRole::Write, plan.element_bytes,
                     plan.output_count));
  return 0;
}

}  // namespace

int RunGraphSignatureContract() {
  if (test_map_signature_builds_without_execution_metadata_payload() != 0) {
    return 1;
  }
  if (test_scatter_signature_closes_indexed_move() != 0) {
    return 1;
  }
  if (RunBoundedGraphSignatureContract() != 0) {
    return 1;
  }
  if (test_factor_signature_closes_lu_outputs() != 0) {
    return 1;
  }
  if (test_solve_signature_closes_factor_reuse_inputs() != 0) {
    return 1;
  }
  return test_spectrum_signature_closes_vector_output();
}

}  // namespace program_compute_contract
