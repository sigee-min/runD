#pragma once

#include "base.hpp"

namespace program_compute_contract::lowering_support {

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
