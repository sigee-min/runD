#include "contract/program/compute/dsl/local.hpp"
#include "test/assert.hpp"

#include <kernel/program/compute/lowering/parse.hpp>

#include <string_view>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

int test_compute_def_builds_canonical_ir_identity() {
  const auto op = BuildIntegrateOp(7);

  TEST_ASSERT(op.ok());
  TEST_ASSERT(std::string_view{op.reason()} == "ok");
  TEST_ASSERT(op.ir().ok);
  TEST_ASSERT(op.ir().scalar == rund::kernel::ComputeScalar::Lane32);
  TEST_ASSERT(op.ir().op_hash_hi != 0u || op.ir().op_hash_lo != 0u);
  TEST_ASSERT(!op.ir().canonical_bytes.empty());
  TEST_ASSERT(op.map().op_hash_hi == op.ir().op_hash_hi);
  TEST_ASSERT(op.map().op_hash_lo == op.ir().op_hash_lo);
  TEST_ASSERT(op.map().scalar == rund::kernel::ComputeScalar::Lane32);
  TEST_ASSERT(op.map().input_bytes_per_tile == 8u);
  TEST_ASSERT(op.map().output_bytes_per_tile == 4u);
  TEST_ASSERT(op.map().param_bytes == 4u);
  TEST_ASSERT(op.map().metadata_bytes_per_tile != 0u);
  return 0;
}

int test_compute_repeated_equivalent_expressions_have_same_hash() {
  const auto first = BuildIntegrateOp(7);
  const auto second = BuildIntegrateOp(7);

  TEST_ASSERT(first.ok());
  TEST_ASSERT(second.ok());
  TEST_ASSERT(first.ir().op_hash_hi == second.ir().op_hash_hi);
  TEST_ASSERT(first.ir().op_hash_lo == second.ir().op_hash_lo);
  TEST_ASSERT(first.ir().canonical_bytes == second.ir().canonical_bytes);
  return 0;
}

int test_compute_uniform_read_has_distinct_canonical_identity() {
  i32 input[1]{};
  i32 output[4]{};
  const auto body =
      rund::compute_dsl::bind(4u).i32().read<"uniform">(input).write<"output">(
          output);
  const auto direct =
      rund::compute_dsl::def("direct-i32")
          .on(body)
          .map([](auto index, auto bindings) {
            const auto source = bindings.template read<"uniform">();
            const auto target = bindings.template write<"output">();
            target[index] = source[index];
          });
  const rund::kernel::ComputeIR uniform = BuildI32UniformReadIr();
  const auto parsed =
      rund::kernel::compute_lowering_detail::ParseComputeIR(uniform);

  TEST_ASSERT(direct.ok());
  TEST_ASSERT(uniform.ok);
  TEST_ASSERT(parsed.ok);
  TEST_ASSERT(parsed.nodes.size() == 2u);
  TEST_ASSERT(static_cast<rund::kernel::IrOp>(parsed.nodes[0].op) ==
              rund::kernel::IrOp::ReadUniform);
  TEST_ASSERT(parsed.nodes[0].lhs == 0u && parsed.nodes[0].rhs == 0u);
  TEST_ASSERT(uniform.canonical_bytes != direct.ir().canonical_bytes);
  TEST_ASSERT(uniform.op_hash_hi != direct.ir().op_hash_hi ||
              uniform.op_hash_lo != direct.ir().op_hash_lo);
  return 0;
}

int test_compute_different_parameter_values_change_hash() {
  const auto first = BuildIntegrateOp(7);
  const auto second = BuildIntegrateOp(8);

  TEST_ASSERT(first.ok());
  TEST_ASSERT(second.ok());
  TEST_ASSERT(first.ir().op_hash_hi != second.ir().op_hash_hi ||
              first.ir().op_hash_lo != second.ir().op_hash_lo);
  TEST_ASSERT(first.ir().canonical_bytes != second.ir().canonical_bytes);
  return 0;
}

int test_compute_diagnostic_names_do_not_change_identity() {
  const auto body = BuildIntegrateBody(7);
  const auto first =
      rund::compute_dsl::def("diagnostic-a").on(body).map([](auto i, auto b) {
        const auto dt = b.template param<"dt">();
        const auto pos = b.template read<"pos">();
        const auto vel = b.template read<"vel">();
        const auto out = b.template write<"out">();
        out[i] = pos[i] + vel[i] * dt;
      });
  const auto second =
      rund::compute_dsl::def("diagnostic-b").on(body).map([](auto i, auto b) {
        const auto dt = b.template param<"dt">();
        const auto pos = b.template read<"pos">();
        const auto vel = b.template read<"vel">();
        const auto out = b.template write<"out">();
        out[i] = pos[i] + vel[i] * dt;
      });

  TEST_ASSERT(first.ok());
  TEST_ASSERT(second.ok());
  TEST_ASSERT(first.ir().canonical_bytes == second.ir().canonical_bytes);
  TEST_ASSERT(first.ir().op_hash_hi == second.ir().op_hash_hi);
  TEST_ASSERT(first.ir().op_hash_lo == second.ir().op_hash_lo);
  return 0;
}

int test_compute_numeric_binding_mode_changes_identity() {
  i32 input[4]{};
  i32 output[4]{};
  const auto signed_body =
      rund::compute_dsl::bind(4u).i32().read<"input">(input).write<"output">(
          output);
  const auto unsigned_body =
      rund::compute_dsl::bind(4u).u32().read<"input">(input).write<"output">(
          output);
  const auto signed_op = rund::compute_dsl::def("same-diagnostic")
                             .on(signed_body)
                             .map([](auto i, auto b) {
                               const auto input = b.template read<"input">();
                               const auto output = b.template write<"output">();
                               output[i] = input[i];
                             });
  const auto unsigned_op = rund::compute_dsl::def("same-diagnostic")
                               .on(unsigned_body)
                               .map([](auto i, auto b) {
                                 const auto input = b.template read<"input">();
                                 const auto output =
                                     b.template write<"output">();
                                 output[i] = input[i];
                               });

  TEST_ASSERT(signed_op.ok());
  TEST_ASSERT(unsigned_op.ok());
  TEST_ASSERT(signed_op.ir().canonical_bytes !=
              unsigned_op.ir().canonical_bytes);
  TEST_ASSERT(signed_op.ir().op_hash_hi != unsigned_op.ir().op_hash_hi ||
              signed_op.ir().op_hash_lo != unsigned_op.ir().op_hash_lo);
  return 0;
}

} // namespace

int RunComputeDslBasicIdentityContract() {
  if (test_compute_def_builds_canonical_ir_identity() != 0) {
    return 1;
  }
  if (test_compute_repeated_equivalent_expressions_have_same_hash() != 0) {
    return 1;
  }
  if (test_compute_uniform_read_has_distinct_canonical_identity() != 0) {
    return 1;
  }
  if (test_compute_different_parameter_values_change_hash() != 0) {
    return 1;
  }
  if (test_compute_diagnostic_names_do_not_change_identity() != 0) {
    return 1;
  }
  return test_compute_numeric_binding_mode_changes_identity();
}

} // namespace program_compute_contract
