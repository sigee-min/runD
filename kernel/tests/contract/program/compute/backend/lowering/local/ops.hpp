#pragma once

[[nodiscard]] inline auto BuildI32DivideOp() {
  i32 input[1]{};
  i32 output[1]{};
  auto body = rund::compute_dsl::bind(1u)
                  .i32()
                  .param<"divisor">(i32{2})
                  .read<"input">(input)
                  .write<"output">(output);
  return rund::compute_dsl::def("divide-i32")
      .on(body)
      .map([](auto index, auto bindings) {
        auto input = bindings.template read<"input">();
        auto output = bindings.template write<"output">();
        output[index] = input[index] / bindings.template param<"divisor">();
      });
}

[[nodiscard]] inline auto BuildU64DivideOp() {
  rund::kernel::u64 input[1]{};
  rund::kernel::u64 output[1]{};
  auto body = rund::compute_dsl::bind(1u)
                  .u64()
                  .param<"divisor">(rund::kernel::u64{2})
                  .read<"input">(input)
                  .write<"output">(output);
  return rund::compute_dsl::def("divide-u64")
      .on(body)
      .map([](auto index, auto bindings) {
        auto input = bindings.template read<"input">();
        auto output = bindings.template write<"output">();
        output[index] = input[index] / bindings.template param<"divisor">();
      });
}

[[nodiscard]] inline auto BuildU64IndexOp() {
  rund::kernel::u64 output[4]{};
  auto body = rund::compute_dsl::bind(4u).u64().write<"output">(output);
  return rund::compute_dsl::def("index-u64")
      .on(body)
      .map([](auto index, auto bindings) {
        auto output = bindings.template write<"output">();
        output[index] = bindings.index();
      });
}

[[nodiscard]] inline auto BuildU64MaskOp() {
  rund::kernel::u64 input[4]{};
  rund::kernel::u32 output[4]{};
  auto body =
      rund::compute_dsl::bind(4u).u64().read<"input">(input).write<"output">(
          output);
  return rund::compute_dsl::def("mask-u64")
      .on(body)
      .map([](auto index, auto bindings) {
        auto input = bindings.template read<"input">();
        auto output = bindings.template write<"output">();
        output[index] = rund::compute_dsl::select(
            rund::compute_dsl::ne(input[index], rund::kernel::u64{0}),
            rund::kernel::u64{1}, rund::kernel::u64{0});
      });
}

[[nodiscard]] inline auto BuildU32MaskOp() {
  rund::kernel::u32 input[4]{};
  rund::kernel::u64 output[4]{};
  auto body =
      rund::compute_dsl::bind(4u).u32().read<"input">(input).write<"output">(
          output);
  return rund::compute_dsl::def("mask-u32")
      .on(body)
      .map([](auto index, auto bindings) {
        auto input = bindings.template read<"input">();
        auto output = bindings.template write<"output">();
        output[index] = rund::compute_dsl::select(
            rund::compute_dsl::ne(input[index], rund::kernel::u32{0}),
            rund::kernel::u32{1}, rund::kernel::u32{0});
      });
}
