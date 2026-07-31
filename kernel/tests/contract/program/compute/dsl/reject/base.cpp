#include "contract/program/compute/dsl/reject/local.hpp"
#include "test/assert.hpp"

#include <string_view>
#include <utility>

namespace program_compute_contract::dsl_reject {
namespace {

using rund::kernel::i32;

template <class T>
concept HasF32Selector = requires(T builder) { builder.f32(); };

template <class T>
concept HasF64Selector = requires(T builder) { builder.f64(); };

using UnboundBuilder = decltype(rund::compute_dsl::bind(1u));
static_assert(!HasF32Selector<UnboundBuilder>);
static_assert(!HasF64Selector<UnboundBuilder>);
static_assert(
    static_cast<rund::kernel::u8>(rund::compute_dsl::detail::ToComputeScalar(
        rund::compute_dsl::detail::ScalarMode::Unspecified)) == 0u);
static_assert(
    static_cast<rund::kernel::u8>(rund::compute_dsl::detail::ToComputeDomain(
        rund::compute_dsl::detail::ScalarMode::Unspecified)) == 0u);

int NumericMode() {
  i32 output[1]{};
  int mapper_calls = 0;
  const auto body =
      rund::compute_dsl::bind(1u).param<"zero">(0).write<"output">(output);
  const auto op = rund::compute_dsl::def("unspecified-mode")
                      .on(body)
                      .map([&](auto index, auto bindings) {
                        ++mapper_calls;
                        auto out = bindings.template write<"output">();
                        out[index] = bindings.template param<"zero">();
                      });

  TEST_ASSERT(!op.ok());
  TEST_ASSERT(std::string_view{op.reason()} == "compute_scalar_unsupported");
  TEST_ASSERT(mapper_calls == 0);

  const auto delayed_mode =
      rund::compute_dsl::bind(1u).write<"output">(output).i32();
  TEST_ASSERT(delayed_mode.bindings().size() == 1u);
  TEST_ASSERT(delayed_mode.bindings().front().numeric_mode ==
              rund::compute_dsl::detail::ScalarMode::I32);
  return 0;
}

int MissingWrite() {
  i32 pos[4]{};
  i32 out[4]{};
  auto body =
      rund::compute_dsl::bind(4u).fixed<1, 31>().read<"pos">(pos).write<"out">(
          out);

  const auto op =
      rund::compute_dsl::def("missing-write").on(body).map([](auto i, auto b) {
        auto pos = b.template read<"pos">();
        (void)pos[i];
      });

  TEST_ASSERT(!op.ok());
  TEST_ASSERT(std::string_view{op.reason()} == "compute_write_missing");
  return 0;
}

int Scatter() {
  i32 pos[4]{};
  i32 out[4]{};
  auto body = rund::compute_dsl::bind(4u)
                  .fixed<1, 31>()
                  .param<"j">(1)
                  .read<"pos">(pos)
                  .write<"out">(out);

  const auto op =
      rund::compute_dsl::def("scatter").on(body).map([](auto i, auto b) {
        auto j = b.template param<"j">();
        auto pos = b.template read<"pos">();
        auto out = b.template write<"out">();
        out[j] = pos[i];
      });

  TEST_ASSERT(!op.ok());
  TEST_ASSERT(std::string_view{op.reason()} ==
              "compute_write_index_unsupported");
  return 0;
}

int BindingId() {
  i32 input[1]{};
  i32 output[1]{};
  const auto body = rund::compute_dsl::bind(1u)
                        .fixed<1, 31>()
                        .param<"p">(1)
                        .read<"input">(input)
                        .write<"output">(output);

  rund::compute_dsl::detail::BuildContext read_context{body.bindings()};
  (void)read_context.read_node(99u);
  TEST_ASSERT(!read_context.ok());
  TEST_ASSERT(std::string_view{read_context.reason()} ==
              "compute_binding_invalid");

  rund::compute_dsl::detail::BuildContext write_context{
      body.bindings(), body.scalar_mode(), body.fixed_format()};
  const rund::kernel::u32 value = write_context.param_node("p");
  write_context.write_node(1u, value);
  TEST_ASSERT(!write_context.ok());
  TEST_ASSERT(std::string_view{write_context.reason()} ==
              "compute_binding_invalid");
  return 0;
}

int FloatParam() {
  i32 output[1]{};
  int mapper_calls = 0;
  const auto body = rund::compute_dsl::bind(1u)
                        .fixed<1, 31>()
                        .param<"dt">(0.5f)
                        .write<"output">(output);

  const auto op =
      rund::compute_dsl::def("float-param").on(body).map([&](auto i, auto b) {
        ++mapper_calls;
        auto dt = b.template param<"dt">();
        auto out = b.template write<"output">();
        out[i] = dt;
      });

  TEST_ASSERT(!op.ok());
  TEST_ASSERT(std::string_view{op.reason()} ==
              "compute_param_float_unsupported");
  TEST_ASSERT(mapper_calls == 0);
  return 0;
}

int InvalidBody() {
  i32 output[1]{};
  int mapper_calls = 0;
  const auto body = rund::compute_dsl::bind(1u)
                        .fixed<1, 31>()
                        .param<"p">(1)
                        .param<"p">(2)
                        .write<"output">(output);
  const auto op = rund::compute_dsl::def("duplicate-no-call")
                      .on(body)
                      .map([&](auto i, auto b) {
                        ++mapper_calls;
                        (void)i;
                        (void)b;
                      });

  TEST_ASSERT(!op.ok());
  TEST_ASSERT(std::string_view{op.reason()} == "compute_binding_duplicate");
  TEST_ASSERT(mapper_calls == 0);
  return 0;
}

} // namespace

int Base() {
  if (NumericMode() != 0 || MissingWrite() != 0 || Scatter() != 0 ||
      BindingId() != 0 || FloatParam() != 0) {
    return 1;
  }
  return InvalidBody();
}

} // namespace program_compute_contract::dsl_reject
