#pragma once

#include <kernel/program/compute/dsl/bind.hpp>
#include <kernel/program/compute/dsl/build.hpp>
#include <kernel/program/compute/dsl/expression/access.hpp>
#include <kernel/program/compute/dsl/operation.hpp>

#include <functional>
#include <string>
#include <utility>

namespace rund::compute_dsl {

using ComputeIndex = detail::Index;
using ComputeValue = detail::Expr;

template <rund::kernel::u8 IntegerBits, rund::kernel::u8 FractionBits,
          rund::kernel::ComputeRounding Round =
              rund::kernel::ComputeRounding::NearestEven,
          rund::kernel::ComputeOverflow Overflow =
              rund::kernel::ComputeOverflow::Saturate,
          rund::kernel::ComputeApproximation Approximation =
              rund::kernel::ComputeApproximation::Exact>
  requires(IntegerBits != 0u && FractionBits != 0u &&
           (static_cast<unsigned>(IntegerBits) + FractionBits == 32u ||
            static_cast<unsigned>(IntegerBits) + FractionBits == 64u))
[[nodiscard]] inline ComputeValue quantize(const ComputeValue value) noexcept {
  return detail::UnaryFormatted(rund::kernel::IrOp::Quantize, value,
                                rund::kernel::ComputeFixedFormat{
                                    .integer_bits = IntegerBits,
                                    .fraction_bits = FractionBits,
                                    .rounding = Round,
                                    .overflow = Overflow,
                                    .approximation = Approximation,
                                });
}

template <typename Body> class ComputeDefBody {
public:
  ComputeDefBody(std::string name, Body body)
      : name_(std::move(name)), body_(std::move(body)) {}

  template <typename Mapper>
    requires std::invocable<Mapper &, ComputeIndex, detail::Access<Body>>
  [[nodiscard]] ComputeOp map(Mapper mapper) const {
    detail::BuildContext context{body_.bindings(), Body::scalar_mode(),
                                 body_.fixed_format()};
    if (!body_.ok() || !detail::NumericMode(Body::scalar_mode()) ||
        detail::HasFloatingPointParam(body_.bindings())) {
      rund::kernel::ComputeIR ir = detail::BuildIr(name_, body_, context);
      rund::kernel::ComputeMap map = detail::BuildMap(ir, body_);
      map.op_hash_hi = 0u;
      map.op_hash_lo = 0u;
      return ComputeOp{std::move(ir), map, body_.bindings(),
                       body_.tile_count()};
    }

    detail::Access<Body> access{context};
    std::invoke(mapper, ComputeIndex{}, access);

    rund::kernel::ComputeIR ir = detail::BuildIr(name_, body_, context);
    rund::kernel::ComputeMap map = detail::BuildMap(ir, body_);
    if (!ir.ok) {
      map.op_hash_hi = 0u;
      map.op_hash_lo = 0u;
    }
    return ComputeOp{std::move(ir), map, body_.bindings(), body_.tile_count()};
  }

private:
  std::string name_;
  Body body_;
};

class ComputeDef {
public:
  explicit ComputeDef(std::string name) : name_(std::move(name)) {}

  template <typename Body>
  [[nodiscard]] ComputeDefBody<Body> on(Body body) const {
    return ComputeDefBody<Body>{name_, std::move(body)};
  }

private:
  std::string name_;
};

} // namespace rund::compute_dsl
