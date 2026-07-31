#include "contract/program/compute/dsl/reject/local.hpp"
#include "test/assert.hpp"

#include <algorithm>
#include <array>
#include <string_view>

namespace program_compute_contract::dsl_reject {
namespace {

using namespace rund::compute_dsl::detail;
using rejection_support::Body;
using rejection_support::Mode;

template <Mode Header>
[[nodiscard]] bool RejectsDomain(const Mode binding_mode,
                                 const BindingKind role) {
  const rund::kernel::u32 bytes = WideMode(Header) ? 8u : 4u;
  const BindingKind source_kind =
      role == BindingKind::Param ? BindingKind::Param : BindingKind::Read;
  Body<Header> body{
      .values = {
          BindingRuntime{
              .kind = source_kind,
              .numeric_mode =
                  role == BindingKind::Write ? Header : binding_mode,
              .name = "source",
              .element_bytes = bytes,
          },
          BindingRuntime{
              .kind = BindingKind::Write,
              .numeric_mode =
                  role == BindingKind::Write ? binding_mode : Header,
              .name = "output",
              .element_bytes = bytes,
          },
      }};
  BuildContext context{body.bindings(), Header, body.fixed_format()};
  if (role == BindingKind::Param) {
    (void)context.param_node("source");
  } else {
    const rund::kernel::u32 value = context.read_node(0u);
    if (role == BindingKind::Write && context.ok()) {
      context.write_node(1u, value);
    }
  }
  return !context.ok() &&
         std::string_view{context.reason()} == "compute_value_invalid";
}

template <Mode Header, std::size_t Count>
[[nodiscard]] bool RejectsDomains(const std::array<Mode, Count> &modes) {
  for (const BindingKind role :
       {BindingKind::Param, BindingKind::Read, BindingKind::Write}) {
    for (const auto mode : modes) {
      if (!RejectsDomain<Header>(mode, role)) {
        return false;
      }
    }
  }
  return true;
}

int Cartesian() {
  TEST_ASSERT(
      (RejectsDomains<Mode::FixedLane32>(std::array{Mode::I32, Mode::U32})));
  TEST_ASSERT(
      (RejectsDomains<Mode::FixedLane64>(std::array{Mode::I64, Mode::U64})));
  TEST_ASSERT((RejectsDomains<Mode::I32>(std::array{Mode::FixedLane32})));
  TEST_ASSERT((RejectsDomains<Mode::U32>(std::array{Mode::FixedLane32})));
  TEST_ASSERT((RejectsDomains<Mode::I64>(std::array{Mode::FixedLane64})));
  TEST_ASSERT((RejectsDomains<Mode::U64>(std::array{Mode::FixedLane64})));
  return 0;
}

[[nodiscard]] Body<Mode::I32> MixedBody() {
  return Body<Mode::I32>{.values = {
                             BindingRuntime{
                                 .kind = BindingKind::Read,
                                 .numeric_mode = Mode::I32,
                                 .name = "signed_input",
                                 .element_bytes = 4u,
                             },
                             BindingRuntime{
                                 .kind = BindingKind::Read,
                                 .numeric_mode = Mode::U32,
                                 .name = "unsigned_input",
                                 .element_bytes = 4u,
                             },
                             BindingRuntime{
                                 .kind = BindingKind::Write,
                                 .numeric_mode = Mode::I32,
                                 .name = "signed_output",
                                 .element_bytes = 4u,
                             },
                             BindingRuntime{
                                 .kind = BindingKind::Write,
                                 .numeric_mode = Mode::U32,
                                 .name = "unsigned_output",
                                 .element_bytes = 4u,
                             },
                         }};
}

struct LiteralEvidence final {
  rund::kernel::ComputeIR ir;
  std::size_t constants = 0u;
  bool modes_match = false;
};

[[nodiscard]] LiteralEvidence BuildLiterals() {
  const auto body = MixedBody();
  BuildContext context{body.bindings(), Mode::I32};
  const auto signed_value = DynamicRead(context, 0u);
  const auto unsigned_value = DynamicRead(context, 1u);
  const auto signed_one = TypedConstant(signed_value, Mode::I32, 1u);
  const auto unsigned_one = TypedConstant(signed_value, Mode::U32, 1u);
  const auto signed_sum =
      Binary(rund::kernel::IrOp::Add, signed_value, signed_one);
  const auto unsigned_sum =
      Binary(rund::kernel::IrOp::Add, unsigned_value, unsigned_one);
  DynamicWrite(context, 2u, signed_sum);
  DynamicWrite(context, 3u, unsigned_sum);
  return LiteralEvidence{
      .ir = BuildIr("", body, context),
      .constants = static_cast<std::size_t>(
          std::count_if(context.nodes().begin(), context.nodes().end(),
                        [](const auto &node) {
                          return node.op == rund::kernel::IrOp::Constant;
                        })),
      .modes_match = ScalarModeOf(signed_one) == Mode::I32 &&
                     ScalarModeOf(unsigned_one) == Mode::U32,
  };
}

int Literals() {
  const auto first = BuildLiterals();
  const auto second = BuildLiterals();
  TEST_ASSERT(first.ir.ok && second.ir.ok);
  TEST_ASSERT(first.constants == 2u && second.constants == 2u);
  TEST_ASSERT(first.modes_match && second.modes_match);
  TEST_ASSERT(first.ir.canonical_bytes == second.ir.canonical_bytes);
  TEST_ASSERT(first.ir.op_hash_hi == second.ir.op_hash_hi);
  TEST_ASSERT(first.ir.op_hash_lo == second.ir.op_hash_lo);
  TEST_ASSERT(rejection_support::Accepts(first.ir));
  return 0;
}

[[nodiscard]] Body<Mode::I32> OutputBody(const Mode input, const Mode output) {
  return Body<Mode::I32>{.values = {
                             BindingRuntime{
                                 .kind = BindingKind::Read,
                                 .numeric_mode = input,
                                 .name = "input",
                                 .element_bytes = 4u,
                             },
                             BindingRuntime{
                                 .kind = BindingKind::Write,
                                 .numeric_mode = output,
                                 .name = "output",
                                 .element_bytes = 4u,
                             },
                         }};
}

int TypedOutput() {
  const auto body = OutputBody(Mode::I32, Mode::U32);
  BuildContext context{body.bindings(), Mode::I32};
  const auto input = DynamicRead(context, 0u);
  const auto value = TypedConstant(input, Mode::U32, 7u);
  TEST_ASSERT(ScalarModeOf(value) == Mode::U32);
  DynamicWrite(context, 1u, value);
  const auto ir = BuildIr("", body, context);
  TEST_ASSERT(ir.ok);
  TEST_ASSERT(rejection_support::Accepts(ir));
  return 0;
}

int PredicateMask() {
  const auto body = OutputBody(Mode::I32, Mode::U32);
  BuildContext context{body.bindings(), Mode::I32};
  const auto input = DynamicRead(context, 0u);
  const auto predicate = Binary(rund::kernel::IrOp::Eq, input, input);
  const auto one = StorageConstant(predicate, 1u);
  const auto zero = StorageConstant(predicate, 0u);
  const auto mask = Ternary(rund::kernel::IrOp::Select, predicate, one, zero);
  TEST_ASSERT(ScalarModeOf(mask) == Mode::I32);
  DynamicWrite(context, 1u, mask);
  const auto ir = BuildIr("", body, context);
  TEST_ASSERT(ir.ok);
  TEST_ASSERT(rejection_support::Accepts(ir));
  return 0;
}

int PredicateDomain() {
  const auto body = OutputBody(Mode::U32, Mode::I32);
  BuildContext context{body.bindings(), Mode::U32};
  const auto input = DynamicRead(context, 0u);
  const auto predicate = Binary(rund::kernel::IrOp::Eq, input, input);
  const auto one = TypedConstant(input, Mode::I32, 1u);
  const auto zero = TypedConstant(input, Mode::I32, 0u);
  const auto selected =
      Ternary(rund::kernel::IrOp::Select, predicate, one, zero);
  TEST_ASSERT(ScalarModeOf(selected) == Mode::I32);
  DynamicWrite(context, 1u, selected);
  const auto ir = BuildIr("", body, context);
  TEST_ASSERT(ir.ok);
  TEST_ASSERT(rejection_support::Accepts(ir));

  BuildContext malformed{body.bindings(), Mode::U32};
  const auto malformed_input = DynamicRead(malformed, 0u);
  const auto malformed_predicate =
      Binary(rund::kernel::IrOp::Eq, malformed_input, malformed_input);
  const auto signed_one = TypedConstant(malformed_input, Mode::I32, 1u);
  const auto unsigned_zero = TypedConstant(malformed_input, Mode::U32, 0u);
  const auto mixed = Ternary(rund::kernel::IrOp::Select, malformed_predicate,
                             signed_one, unsigned_zero);
  DynamicWrite(malformed, 1u, mixed);
  TEST_ASSERT(!malformed.ok());
  TEST_ASSERT(std::string_view{malformed.reason()} == "compute_value_invalid");
  return 0;
}

} // namespace

int Binding() {
  if (Cartesian() != 0 || Literals() != 0 || TypedOutput() != 0 ||
      PredicateMask() != 0) {
    return 1;
  }
  return PredicateDomain();
}

} // namespace program_compute_contract::dsl_reject
