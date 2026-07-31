#include "contract/program/compute/dsl/local.hpp"
#include "test/assert.hpp"

#include <kernel/program/compute/lowering/parse.hpp>
#include <kernel/program/compute/lowering/validate.hpp>

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
  const auto first = rund::compute_dsl::def("diagnostic-a").on(body).map(
      [](auto i, auto b) {
        const auto dt = b.template param<"dt">();
        const auto pos = b.template read<"pos">();
        const auto vel = b.template read<"vel">();
        const auto out = b.template write<"out">();
        out[i] = pos[i] + vel[i] * dt;
      });
  const auto second = rund::compute_dsl::def("diagnostic-b").on(body).map(
      [](auto i, auto b) {
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
  const auto signed_body = rund::compute_dsl::bind(4u)
                               .i32()
                               .read<"input">(input)
                               .write<"output">(output);
  const auto unsigned_body = rund::compute_dsl::bind(4u)
                                 .u32()
                                 .read<"input">(input)
                                 .write<"output">(output);
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
                                 const auto output = b.template write<"output">();
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

int test_compute_fixed_format_and_policy_change_identity() {
  const auto make = []<rund::kernel::u8 I, rund::kernel::u8 F,
                       rund::kernel::ComputeRounding Round =
                           rund::kernel::ComputeRounding::NearestEven,
                       rund::kernel::ComputeOverflow Overflow =
                           rund::kernel::ComputeOverflow::Saturate,
                       rund::kernel::ComputeApproximation Approximation =
                           rund::kernel::ComputeApproximation::Exact>() {
    i32 input[1]{};
    i32 output[1]{};
    const auto body = rund::compute_dsl::bind(1u)
                          .template fixed<I, F, Round, Overflow,
                                          Approximation>()
                          .template read<"input">(input)
                          .template write<"output">(output);
    return rund::compute_dsl::def("fixed-policy-identity")
        .on(body)
        .map([](auto i, auto b) {
          const auto input = b.template read<"input">();
          const auto output = b.template write<"output">();
          output[i] = input[i];
        });
  };

  const auto baseline = make.template operator()<1u, 31u>();
  const auto integer_fraction = make.template operator()<16u, 16u>();
  const auto rounding =
      make.template operator()<1u, 31u, rund::kernel::ComputeRounding::Down>();
  const auto overflow = make.template operator()<
      1u, 31u, rund::kernel::ComputeRounding::NearestEven,
      rund::kernel::ComputeOverflow::Wrap>();
  const auto approximation = make.template operator()<
      1u, 31u, rund::kernel::ComputeRounding::NearestEven,
      rund::kernel::ComputeOverflow::Saturate,
      rund::kernel::ComputeApproximation::Deterministic>();

  TEST_ASSERT(baseline.ok());
  TEST_ASSERT(integer_fraction.ok());
  TEST_ASSERT(rounding.ok());
  TEST_ASSERT(overflow.ok());
  TEST_ASSERT(approximation.ok());
  const auto differs = [&](const auto &candidate) {
    return baseline.ir().op_hash_hi != candidate.ir().op_hash_hi ||
           baseline.ir().op_hash_lo != candidate.ir().op_hash_lo;
  };
  TEST_ASSERT(differs(integer_fraction));
  TEST_ASSERT(differs(rounding));
  TEST_ASSERT(differs(overflow));
  TEST_ASSERT(differs(approximation));
  return 0;
}

int test_compute_mul_add_fixed_uses_tight_signed_bound() {
  i32 lhs[1]{};
  i32 rhs[1]{};
  i32 addend[1]{};
  i32 output[1]{};
  const auto body = rund::compute_dsl::bind(1u)
                        .fixed<16u, 16u>()
                        .read<"lhs">(lhs)
                        .read<"rhs">(rhs)
                        .read<"addend">(addend)
                        .write<"output">(output);
  const auto op = rund::compute_dsl::def("fixed-mul-add-carry")
                      .on(body)
                      .map([](auto i, auto b) {
                        const auto lhs_values = b.template read<"lhs">();
                        const auto rhs_values = b.template read<"rhs">();
                        const auto addend_values =
                            b.template read<"addend">();
                        const auto output_values =
                            b.template write<"output">();
                        output_values[i] = rund::compute_dsl::quantize<16u, 16u>(
                            rund::compute_dsl::mul_add_fixed(
                                lhs_values[i], rhs_values[i], addend_values[i]));
                      });

  TEST_ASSERT(op.ok());
  auto parsed =
      rund::kernel::compute_lowering_detail::ParseComputeIR(op.ir());
  TEST_ASSERT(parsed.ok);
  bool saw_mul_add = false;
  for (const auto &node : parsed.nodes) {
    if (node.op !=
        static_cast<rund::kernel::u8>(rund::kernel::IrOp::MulAddFixed)) {
      continue;
    }
    saw_mul_add = true;
    TEST_ASSERT(node.fixed_format.integer_bits == 32u);
    TEST_ASSERT(node.fixed_format.fraction_bits == 32u);
  }
  TEST_ASSERT(saw_mul_add);

  auto forged = parsed;
  for (auto &node : forged.nodes) {
    if (node.op ==
        static_cast<rund::kernel::u8>(rund::kernel::IrOp::MulAddFixed)) {
      node.fixed_format.integer_bits = 31u;
    }
  }
  const char *const rejected =
      rund::kernel::compute_lowering_detail::ValidateLowerableIR(
          forged, op.ir().scalar);
  TEST_ASSERT(rejected != nullptr);
  TEST_ASSERT(std::string_view{rejected} ==
              "compute_ir_numeric_policy_mismatch");
  return 0;
}

int test_compute_fixed_identity_write_has_explicit_quantize() {
  i32 input[1]{};
  i32 output[1]{};
  const auto body = rund::compute_dsl::bind(1u)
                        .fixed<16u, 16u>()
                        .read<"input">(input)
                        .write<"output">(output);
  const auto op = rund::compute_dsl::def("fixed-identity-quantize")
                      .on(body)
                      .map([](auto i, auto b) {
                        const auto source = b.template read<"input">();
                        const auto target = b.template write<"output">();
                        target[i] = source[i];
                      });
  TEST_ASSERT(op.ok());
  const auto parsed =
      rund::kernel::compute_lowering_detail::ParseComputeIR(op.ir());
  TEST_ASSERT(parsed.ok);
  bool saw_write = false;
  auto forged = parsed;
  for (const auto &node : parsed.nodes) {
    if (node.op != static_cast<rund::kernel::u8>(rund::kernel::IrOp::Write)) {
      continue;
    }
    saw_write = true;
    TEST_ASSERT(node.lhs != 0u && node.lhs <= parsed.nodes.size());
    const auto &quantize = parsed.nodes[node.lhs - 1u];
    TEST_ASSERT(quantize.op ==
                static_cast<rund::kernel::u8>(rund::kernel::IrOp::Quantize));
    TEST_ASSERT(quantize.lhs != 0u && quantize.lhs <= parsed.nodes.size());
    const auto &read = parsed.nodes[quantize.lhs - 1u];
    TEST_ASSERT(read.op ==
                static_cast<rund::kernel::u8>(rund::kernel::IrOp::Read));
    TEST_ASSERT(node.fixed_format == quantize.fixed_format);
    TEST_ASSERT(quantize.fixed_format.integer_bits == 16u);
    TEST_ASSERT(quantize.fixed_format.fraction_bits == 16u);
  }
  TEST_ASSERT(saw_write);
  for (auto &node : forged.nodes) {
    if (node.op != static_cast<rund::kernel::u8>(rund::kernel::IrOp::Write)) {
      continue;
    }
    const auto &quantize = forged.nodes[node.lhs - 1u];
    node.lhs = quantize.lhs;
    node.fixed_format = forged.nodes[node.lhs - 1u].fixed_format;
  }
  const char *const rejected =
      rund::kernel::compute_lowering_detail::ValidateLowerableIR(
          forged, op.ir().scalar);
  TEST_ASSERT(rejected != nullptr);
  TEST_ASSERT(std::string_view{rejected} == "compute_ir_quantize_required");
  return 0;
}

int test_compute_formatted_constant_owns_numeric_policy() {
  using namespace rund::compute_dsl::detail;
  using namespace rund::kernel;
  i64 input[1]{};
  const auto body = rund::compute_dsl::bind(1u)
                        .fixed<20u, 44u>()
                        .read<"input">(input);
  constexpr ComputeFixedFormat storage{
      .integer_bits = 20u,
      .fraction_bits = 44u,
      .rounding = ComputeRounding::NearestEven,
      .overflow = ComputeOverflow::Saturate,
      .approximation = ComputeApproximation::Exact,
  };
  constexpr ComputeFixedFormat alternate{
      .integer_bits = 21u,
      .fraction_bits = 43u,
      .rounding = ComputeRounding::Down,
      .overflow = ComputeOverflow::Wrap,
      .approximation = ComputeApproximation::Deterministic,
  };
  BuildContext context{body.bindings(), ScalarMode::FixedLane64, storage};
  const auto anchor = DynamicRead(context, 0u, storage);
  (void)FormattedConstant(anchor, 0x123456789abcdef0ull, alternate);
  TEST_ASSERT(context.ok());
  TEST_ASSERT(context.nodes().size() == 2u);
  TEST_ASSERT(context.nodes().back().op == IrOp::Constant);
  TEST_ASSERT(context.nodes().back().fixed_format == alternate);

  BuildContext invalid{body.bindings(), ScalarMode::FixedLane64, storage};
  const auto invalid_anchor = DynamicRead(invalid, 0u, storage);
  (void)FormattedConstant(invalid_anchor, 1u, {});
  TEST_ASSERT(!invalid.ok());
  TEST_ASSERT(std::string_view{invalid.reason()} ==
              "compute_fixed_format_invalid");
  return 0;
}

int test_compute_ternary_rejects_split_precision_over_128_bits() {
  using namespace rund::compute_dsl::detail;
  using namespace rund::kernel;
  i64 input[1]{};
  const auto body = rund::compute_dsl::bind(1u)
                        .fixed<20u, 44u>()
                        .read<"input">(input);
  constexpr ComputeFixedFormat storage{
      .integer_bits = 20u,
      .fraction_bits = 44u,
      .rounding = ComputeRounding::NearestEven,
      .overflow = ComputeOverflow::Saturate,
      .approximation = ComputeApproximation::Exact,
  };
  constexpr ComputeFixedFormat integer_heavy{
      .integer_bits = 126u,
      .fraction_bits = 2u,
      .rounding = ComputeRounding::NearestEven,
      .overflow = ComputeOverflow::Saturate,
      .approximation = ComputeApproximation::Exact,
  };
  constexpr ComputeFixedFormat fraction_heavy{
      .integer_bits = 2u,
      .fraction_bits = 126u,
      .rounding = ComputeRounding::NearestEven,
      .overflow = ComputeOverflow::Saturate,
      .approximation = ComputeApproximation::Exact,
  };

  BuildContext select_context{body.bindings(), ScalarMode::FixedLane64,
                              storage};
  const auto select_anchor = DynamicRead(select_context, 0u, storage);
  const auto select_true =
      FormattedConstant(select_anchor, 1u, integer_heavy);
  const auto select_false =
      FormattedConstant(select_anchor, 2u, fraction_heavy);
  (void)Ternary(IrOp::Select, select_anchor, select_true, select_false);
  TEST_ASSERT(!select_context.ok());
  TEST_ASSERT(std::string_view{select_context.reason()} ==
              "compute_fixed_precision_capacity");

  BuildContext clamp_context{body.bindings(), ScalarMode::FixedLane64,
                             storage};
  const auto clamp_anchor = DynamicRead(clamp_context, 0u, storage);
  const auto clamp_value =
      FormattedConstant(clamp_anchor, 1u, integer_heavy);
  const auto clamp_low =
      FormattedConstant(clamp_anchor, 2u, integer_heavy);
  const auto clamp_high =
      FormattedConstant(clamp_anchor, 3u, fraction_heavy);
  (void)Ternary(IrOp::Clamp, clamp_value, clamp_low, clamp_high);
  TEST_ASSERT(!clamp_context.ok());
  TEST_ASSERT(std::string_view{clamp_context.reason()} ==
              "compute_fixed_precision_capacity");
  return 0;
}

int test_compute_fixed_format_query_is_read_only_and_never_defaults() {
  using namespace rund::compute_dsl::detail;
  using namespace rund::kernel;
  i32 fixed_input[1]{};
  const auto fixed_body = rund::compute_dsl::bind(1u)
                              .fixed<16u, 16u>()
                              .read<"input">(fixed_input);
  BuildContext fixed_context{fixed_body.bindings(), ScalarMode::FixedLane32,
                             fixed_body.fixed_format()};
  const auto fixed_value = DynamicRead(fixed_context, 0u);
  TEST_ASSERT(FixedFormatOf(fixed_value) == fixed_body.fixed_format());
  TEST_ASSERT(ComputeFixedFormatAbsent(fixed_context.fixed_format_node(99u)));
  TEST_ASSERT(fixed_context.ok());

  i32 integer_input[1]{};
  const auto integer_body =
      rund::compute_dsl::bind(1u).i32().read<"input">(integer_input);
  BuildContext integer_context{integer_body.bindings(), ScalarMode::I32};
  const auto integer_value = DynamicRead(integer_context, 0u);
  TEST_ASSERT(ComputeFixedFormatAbsent(FixedFormatOf(integer_value)));
  TEST_ASSERT(ComputeFixedFormatAbsent(FixedFormatOf(Expr{})));
  TEST_ASSERT(integer_context.ok());
  return 0;
}

int test_compute_undeclared_capture_access_is_not_public_builder_shape() {
  const auto body = BuildIntegrateBody(7);

  static_assert(!decltype(body)::template has_read<"missing">());
  static_assert(!decltype(body)::template has_param<"missing">());
  static_assert(!decltype(body)::template has_write<"missing">());
  return 0;
}

}  // namespace

int RunComputeDslIdentityContract() {
  if (test_compute_def_builds_canonical_ir_identity() != 0) {
    return 1;
  }
  if (test_compute_repeated_equivalent_expressions_have_same_hash() != 0) {
    return 1;
  }
  if (test_compute_different_parameter_values_change_hash() != 0) {
    return 1;
  }
  if (test_compute_diagnostic_names_do_not_change_identity() != 0) {
    return 1;
  }
  if (test_compute_numeric_binding_mode_changes_identity() != 0) {
    return 1;
  }
  if (test_compute_fixed_format_and_policy_change_identity() != 0) {
    return 1;
  }
  if (test_compute_mul_add_fixed_uses_tight_signed_bound() != 0) {
    return 1;
  }
  if (test_compute_fixed_identity_write_has_explicit_quantize() != 0) {
    return 1;
  }
  if (test_compute_formatted_constant_owns_numeric_policy() != 0) {
    return 1;
  }
  if (test_compute_ternary_rejects_split_precision_over_128_bits() != 0) {
    return 1;
  }
  if (test_compute_fixed_format_query_is_read_only_and_never_defaults() != 0) {
    return 1;
  }
  return test_compute_undeclared_capture_access_is_not_public_builder_shape();
}

}  // namespace program_compute_contract
