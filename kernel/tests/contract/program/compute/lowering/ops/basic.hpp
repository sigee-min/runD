#pragma once

#include "base.hpp"

namespace program_compute_contract::lowering_support {

[[nodiscard]] inline rund::kernel::ComputeIR BuildI32UniformReadIr() {
  i32 uniform[1]{};
  i32 output[4]{};
  const auto body = rund::compute_dsl::bind(4u)
                        .i32()
                        .read<"uniform">(uniform)
                        .write<"output">(output);
  rund::compute_dsl::detail::BuildContext context{
      body.bindings(), rund::compute_dsl::detail::ScalarMode::I32};
  const rund::kernel::u32 uniform_binding = context.binding_index(
      "uniform", rund::compute_dsl::detail::BindingKind::Read);
  const rund::kernel::u32 output_binding = context.binding_index(
      "output", rund::compute_dsl::detail::BindingKind::Write);
  const auto value =
      rund::compute_dsl::detail::DynamicUniformRead(context, uniform_binding);
  rund::compute_dsl::detail::DynamicWrite(context, output_binding, value);
  return rund::compute_dsl::detail::BuildIr("uniform-i32", body, context);
}

[[nodiscard]] inline auto BuildFixedLane32Op(const i32 dt) {
  i32 pos[4]{};
  i32 vel[4]{};
  i32 out[4]{};
  const auto body = rund::compute_dsl::bind(4u)
                        .fixed<1, 31>()
                        .param<"dt">(dt)
                        .read<"pos">(pos)
                        .read<"vel">(vel)
                        .write<"out">(out);

  return rund::compute_dsl::def("lower-integrate").on(body).map([](auto i, auto b) {
    auto dt_value = b.template param<"dt">();
    auto pos = b.template read<"pos">();
    auto vel = b.template read<"vel">();
    auto out = b.template write<"out">();

    const auto pos_value = pos[i];
    const auto vel_value = vel[i];
    const auto scaled_velocity = vel_value * dt_value;
    out[i] = pos_value + scaled_velocity;
  });
}

[[nodiscard]] inline auto BuildFixedLane64Op(const i64 dt) {
  i64 pos[4]{};
  i64 out[4]{};
  const auto body = rund::compute_dsl::bind(4u)
                        .fixed<1, 63>()
                        .param<"dt">(dt)
                        .read<"pos">(pos)
                        .write<"out">(out);

  return rund::compute_dsl::def("lower-fixed_lane64").on(body).map([](auto i, auto b) {
    auto dt_value = b.template param<"dt">();
    auto pos = b.template read<"pos">();
    auto out = b.template write<"out">();

    out[i] = pos[i] + dt_value;
  });
}

[[nodiscard]] inline auto BuildFixedLane64MulOp(const i64 dt) {
  i64 pos[4]{};
  i64 out[4]{};
  const auto body = rund::compute_dsl::bind(4u)
                        .fixed<1, 63>()
                        .param<"dt">(dt)
                        .read<"pos">(pos)
                        .write<"out">(out);

  return rund::compute_dsl::def("lower-fixed_lane64-mul").on(body).map([](auto i, auto b) {
    auto dt_value = b.template param<"dt">();
    auto pos = b.template read<"pos">();
    auto out = b.template write<"out">();

    out[i] = pos[i] * dt_value;
  });
}

} // namespace program_compute_contract::lowering_support
