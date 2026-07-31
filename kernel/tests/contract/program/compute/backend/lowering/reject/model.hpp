#pragma once

#include "contract/program/compute/backend/lowering/reject/local.hpp"

#include <array>
#include <limits>

namespace program_compute_contract::lowering_reject {

using rejection_support::Mode;

struct IntegerDomain final {
  rund::kernel::ComputeDomain domain;
  rund::kernel::ComputeScalar scalar;
};

struct OpArity final {
  rund::kernel::IrOp op;
  rund::kernel::u32 arity;
};

inline constexpr std::array kIntegerDomains{
    IntegerDomain{rund::kernel::ComputeDomain::I32,
                  rund::kernel::ComputeScalar::Lane32},
    IntegerDomain{rund::kernel::ComputeDomain::U32,
                  rund::kernel::ComputeScalar::Lane32},
    IntegerDomain{rund::kernel::ComputeDomain::I64,
                  rund::kernel::ComputeScalar::Lane64},
    IntegerDomain{rund::kernel::ComputeDomain::U64,
                  rund::kernel::ComputeScalar::Lane64},
};

[[nodiscard]] constexpr bool Signed(const rund::kernel::ComputeDomain domain) {
  return domain == rund::kernel::ComputeDomain::I32 ||
         domain == rund::kernel::ComputeDomain::I64;
}

[[nodiscard]] rund::kernel::ComputeIR
IntegerIr(rund::kernel::IrOp op, rund::kernel::u32 arity, IntegerDomain mode);

[[nodiscard]] bool SetNode(std::vector<rund::kernel::u8> &bytes,
                           rund::kernel::u32 one_based_node,
                           rund::kernel::IrOp op, rund::kernel::u32 lhs,
                           rund::kernel::u32 rhs,
                           rund::kernel::u32 aux) noexcept;
[[nodiscard]] bool SetFirstMode(std::vector<rund::kernel::u8> &bytes,
                                rund::kernel::u8 mode) noexcept;
[[nodiscard]] bool SetMode(std::vector<rund::kernel::u8> &bytes,
                           rund::kernel::u32 target,
                           rund::kernel::u8 mode) noexcept;
[[nodiscard]] rund::kernel::u32
FindNode(const rund::kernel::compute_lowering_detail::ParsedIR &parsed,
         rund::kernel::IrOp op) noexcept;

template <Mode Source, Mode Target>
[[nodiscard]] rejection_support::Body<Source> CarrierBody() {
  using namespace rund::compute_dsl::detail;
  return rejection_support::Body<Source>{
      .values = {
          BindingRuntime{
              .kind = BindingKind::Read,
              .numeric_mode = Source,
              .name = "source",
              .element_bytes = WideMode(Source) ? 8u : 4u,
          },
          BindingRuntime{
              .kind = BindingKind::Write,
              .numeric_mode = Target,
              .name = "output",
              .element_bytes = WideMode(Target) ? 8u : 4u,
          },
      }};
}

template <Mode Source, Mode Target, bool Expression = false>
[[nodiscard]] rund::kernel::ComputeIR CheckedOrdinalIr() {
  using namespace rund::compute_dsl::detail;
  const auto body = CarrierBody<Source, Target>();
  BuildContext context{body.bindings(), Source};
  const auto read = DynamicRead(context, 0u);
  const auto zero = TypedConstant(read, Source, 0u);
  const auto source = [&] {
    if constexpr (Expression) {
      return Binary(rund::kernel::IrOp::Add, read, zero);
    }
    return read;
  }();
  if constexpr (Source == Mode::I32 || Source == Mode::I64) {
    const auto representable = Binary(rund::kernel::IrOp::Ge, source, zero);
    DynamicCheckedOrdinalWrite(
        context, 1u,
        Ternary(rund::kernel::IrOp::Select, representable, source, zero));
  } else {
    constexpr rund::kernel::u64 maximum =
        Source == Mode::U64
            ? static_cast<rund::kernel::u64>(
                  std::numeric_limits<rund::kernel::i64>::max())
            : static_cast<rund::kernel::u64>(
                  std::numeric_limits<rund::kernel::i32>::max());
    const auto limit = TypedConstant(source, Source, maximum);
    const auto representable =
        Binary(rund::kernel::IrOp::LeUnsigned, source, limit);
    DynamicCheckedOrdinalWrite(
        context, 1u,
        Ternary(rund::kernel::IrOp::Select, representable, source, zero));
  }
  return BuildIr("checked-ordinal", body, context);
}

template <Mode Source, Mode Target> [[nodiscard]] bool CarrierRejects() {
  using namespace rund::compute_dsl::detail;
  const auto body = CarrierBody<Source, Target>();
  BuildContext context{body.bindings(), Source};
  DynamicWrite(context, 1u, DynamicRead(context, 0u));
  return !context.ok() &&
         std::string_view{context.reason()} == "compute_value_invalid";
}

template <Mode Source, Mode Target, bool Expression = false>
[[nodiscard]] rund::kernel::ComputeIR
BoundaryIr(const rund::kernel::ComputeFixedFormat format) {
  using namespace rund::compute_dsl::detail;
  const auto body = CarrierBody<Source, Target>();
  BuildContext context{body.bindings(), Source};
  const auto read = DynamicRead(context, 0u);
  const auto zero = TypedConstant(read, Source, 0u);
  const auto source = [&] {
    if constexpr (Expression) {
      return Binary(rund::kernel::IrOp::Add, read, zero);
    }
    return read;
  }();
  const auto predicate = Binary(rund::kernel::IrOp::Ne, source, zero);
  const auto one = TypedConstant(source, Source, 1u);
  const auto mask = Ternary(rund::kernel::IrOp::Select, predicate, one, zero);
  DynamicBoundaryMaskWrite(context, 1u, mask, format);
  return BuildIr("boundary-mask", body, context);
}

template <Mode Source, Mode Target>
[[nodiscard]] rund::kernel::ComputeIR ValueMaskIr() {
  using namespace rund::compute_dsl::detail;
  const auto body = CarrierBody<Source, Target>();
  BuildContext context{body.bindings(), Source};
  const auto source = DynamicRead(context, 0u);
  const auto zero = TypedConstant(source, Source, 0u);
  const auto predicate = Binary(rund::kernel::IrOp::Ne, source, zero);
  const auto one = TypedConstant(source, Source, 1u);
  DynamicWrite(context, 1u,
               Ternary(rund::kernel::IrOp::Select, predicate, one, zero));
  return BuildIr("boundary-mask", body, context);
}

template <Mode Source, Mode Target> [[nodiscard]] bool BoundaryRejects() {
  const auto ir = ValueMaskIr<Source, Target>();
  return !ir.ok && std::string_view{ir.reason} == "compute_value_invalid";
}

template <Mode Source, Mode Target>
[[nodiscard]] bool
FormatRejects(const rund::kernel::ComputeFixedFormat format) {
  const auto ir = BoundaryIr<Source, Target>(format);
  return !ir.ok && std::string_view{ir.reason} == "compute_value_invalid";
}

template <Mode Header>
[[nodiscard]] rund::kernel::ComputeIR
ValidBindingIr(const rund::compute_dsl::detail::BindingKind source_kind) {
  using namespace rund::compute_dsl::detail;
  const rund::kernel::u32 bytes = WideMode(Header) ? 8u : 4u;
  BindingRuntime source{
      .kind = source_kind,
      .numeric_mode = Header,
      .name = "source",
      .element_bytes = bytes,
  };
  if (source_kind == BindingKind::Param) {
    source.value_bytes.assign(bytes, rund::kernel::u8{0u});
  }
  rejection_support::Body<Header> body{.values = {
                                           std::move(source),
                                           BindingRuntime{
                                               .kind = BindingKind::Write,
                                               .numeric_mode = Header,
                                               .name = "output",
                                               .element_bytes = bytes,
                                           },
                                       }};
  BuildContext context{body.bindings(), Header, body.fixed_format()};
  const rund::kernel::u32 value = source_kind == BindingKind::Param
                                      ? context.param_node("source")
                                      : context.read_node(0u);
  context.write_node(1u, value);
  return BuildIr("", body, context);
}

} // namespace program_compute_contract::lowering_reject
