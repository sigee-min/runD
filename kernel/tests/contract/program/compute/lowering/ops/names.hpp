#pragma once

#include "base.hpp"

namespace program_compute_contract::lowering_support {

[[nodiscard]] inline auto BuildUnsafeNameOp(const i32 dt) {
  i32 pos[2]{};
  i32 out[2]{};
  const auto body = rund::compute_dsl::bind(2u)
                        .fixed<1, 31>()
                        .param<"dt-ms">(dt)
                        .read<"pos bad\nx">(pos)
                        .write<"out!">(out);

  return rund::compute_dsl::def("op bad\nname!").on(body).map([](auto i, auto b) {
    auto dt_value = b.template param<"dt-ms">();
    auto pos = b.template read<"pos bad\nx">();
    auto out = b.template write<"out!">();

    out[i] = pos[i] + dt_value;
  });
}

[[nodiscard]] inline auto BuildCollisionNameOp() {
  i32 dash[2]{};
  i32 escape[2]{};
  i32 digit[2]{};
  i32 prefixed[2]{};
  i32 out[2]{};
  const auto body = rund::compute_dsl::bind(2u)
                        .fixed<1, 31>()
                        .read<"a-b">(dash)
                        .read<"a_x2d_b">(escape)
                        .read<"1">(digit)
                        .read<"rund_1">(prefixed)
                        .write<"out">(out);

  return rund::compute_dsl::def("collision-lowering")
      .on(body)
      .map([](auto i, auto b) {
        auto dash_value = b.template read<"a-b">();
        auto escape_value = b.template read<"a_x2d_b">();
        auto digit_value = b.template read<"1">();
        auto prefixed_value = b.template read<"rund_1">();
        auto write = b.template write<"out">();

        write[i] = dash_value[i] + escape_value[i] + digit_value[i] +
                   prefixed_value[i];
      });
}

} // namespace program_compute_contract::lowering_support
