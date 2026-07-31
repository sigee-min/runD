#include "contract/program/compute/dsl/local.hpp"
#include "test/assert.hpp"

namespace program_compute_contract {
namespace {

using namespace dsl_support;

int test_compute_escaped_values_do_not_mutate_new_context_at_same_address() {
  i32 output[1]{};
  const auto body =
      rund::compute_dsl::bind(1u).fixed<1, 31>().param<"p">(1).write<"output">(output);

  using Context = rund::compute_dsl::detail::BuildContext;
  alignas(Context) unsigned char storage[sizeof(Context)]{};
  rund::compute_dsl::ComputeValue escaped;

  auto* first = new (storage) Context{body.bindings()};
  {
    rund::compute_dsl::detail::Access<decltype(body)> access{*first};
    escaped = access.template param<"p">();
  }
  first->~Context();

  auto* second = new (storage) Context{body.bindings()};
  (void)(escaped + escaped);
  TEST_ASSERT(second->ok());
  TEST_ASSERT(second->nodes().empty());
  second->~Context();
  return 0;
}

int test_compute_escaped_read_handle_does_not_mutate_new_context_at_same_address() {
  i32 input[1]{};
  const auto body = rund::compute_dsl::bind(1u).fixed<1, 31>().read<"input">(input);

  using Context = rund::compute_dsl::detail::BuildContext;
  alignas(Context) unsigned char storage[sizeof(Context)]{};
  std::optional<rund::compute_dsl::detail::ReadHandle> escaped;

  auto* first = new (storage) Context{body.bindings()};
  {
    rund::compute_dsl::detail::Access<decltype(body)> access{*first};
    escaped.emplace(access.template read<"input">());
  }
  first->~Context();

  auto* second = new (storage) Context{body.bindings()};
  (void)((*escaped)[rund::compute_dsl::ComputeIndex{}]);
  TEST_ASSERT(second->ok());
  TEST_ASSERT(second->nodes().empty());
  second->~Context();
  return 0;
}

int test_compute_escaped_write_handle_does_not_mutate_new_context_at_same_address() {
  i32 output[1]{};
  const auto body =
      rund::compute_dsl::bind(1u).fixed<1, 31>().param<"p">(1).write<"output">(output);

  using Context = rund::compute_dsl::detail::BuildContext;
  alignas(Context) unsigned char storage[sizeof(Context)]{};
  std::optional<rund::compute_dsl::detail::WriteHandle> escaped;

  auto* first = new (storage) Context{body.bindings()};
  {
    rund::compute_dsl::detail::Access<decltype(body)> access{*first};
    escaped.emplace(access.template write<"output">());
  }
  first->~Context();

  auto* second = new (storage) Context{body.bindings()};
  rund::compute_dsl::detail::Access<decltype(body)> access{*second};
  auto live_value = access.template param<"p">();
  (*escaped)[rund::compute_dsl::ComputeIndex{}] = live_value;
  TEST_ASSERT(second->ok());
  TEST_ASSERT(second->write_count() == 0u);
  TEST_ASSERT(second->nodes().size() == 1u);
  second->~Context();
  return 0;
}

int test_compute_escaped_write_target_does_not_mutate_new_context_at_same_address() {
  i32 output[1]{};
  const auto body =
      rund::compute_dsl::bind(1u).fixed<1, 31>().param<"p">(1).write<"output">(output);

  using Context = rund::compute_dsl::detail::BuildContext;
  alignas(Context) unsigned char storage[sizeof(Context)]{};
  std::optional<rund::compute_dsl::detail::WriteTarget> escaped;

  auto* first = new (storage) Context{body.bindings()};
  {
    rund::compute_dsl::detail::Access<decltype(body)> access{*first};
    escaped.emplace(
        access.template write<"output">()[rund::compute_dsl::ComputeIndex{}]);
  }
  first->~Context();

  auto* second = new (storage) Context{body.bindings()};
  rund::compute_dsl::detail::Access<decltype(body)> access{*second};
  auto live_value = access.template param<"p">();
  std::move(*escaped) = live_value;
  TEST_ASSERT(second->ok());
  TEST_ASSERT(second->write_count() == 0u);
  TEST_ASSERT(second->nodes().size() == 1u);
  second->~Context();
  return 0;
}

int test_compute_escaped_access_does_not_mutate_new_context_at_same_address() {
  i32 input[1]{};
  i32 output[1]{};
  const auto body = rund::compute_dsl::bind(1u)
                        .fixed<1, 31>()
                        .param<"p">(1)
                        .read<"input">(input)
                        .write<"output">(output);

  using Context = rund::compute_dsl::detail::BuildContext;
  using Body = std::remove_cv_t<decltype(body)>;
  using Access = rund::compute_dsl::detail::Access<Body>;
  alignas(Context) unsigned char storage[sizeof(Context)]{};
  std::optional<Access> escaped;

  auto* first = new (storage) Context{body.bindings()};
  escaped.emplace(*first);
  first->~Context();

  auto* second = new (storage) Context{body.bindings()};
  auto escaped_value = escaped->template param<"p">();
  auto escaped_read = escaped->template read<"input">();
  auto escaped_write = escaped->template write<"output">();
  (void)escaped_read[rund::compute_dsl::ComputeIndex{}];
  escaped_write[rund::compute_dsl::ComputeIndex{}] = escaped_value;
  TEST_ASSERT(second->ok());
  TEST_ASSERT(second->nodes().empty());
  TEST_ASSERT(second->write_count() == 0u);
  second->~Context();

  std::optional<Access> escaped_from_mapper;
  const auto op = rund::compute_dsl::def("escape-access").on(body).map(
      [&](auto i, auto b) {
        escaped_from_mapper.emplace(b);
        auto value = b.template param<"p">();
        auto read = b.template read<"input">();
        auto write = b.template write<"output">();
        write[i] = value + read[i];
      });
  TEST_ASSERT(op.ok());
  TEST_ASSERT(escaped_from_mapper.has_value());

  auto escaped_map_value = escaped_from_mapper->template param<"p">();
  auto escaped_map_read = escaped_from_mapper->template read<"input">();
  auto escaped_map_write = escaped_from_mapper->template write<"output">();
  (void)escaped_map_read[rund::compute_dsl::ComputeIndex{}];
  escaped_map_write[rund::compute_dsl::ComputeIndex{}] = escaped_map_value;
  return 0;
}

}  // namespace

int RunComputeDslEscapeContract() {
  if (test_compute_escaped_values_do_not_mutate_new_context_at_same_address() !=
      0) {
    return 1;
  }
  if (test_compute_escaped_read_handle_does_not_mutate_new_context_at_same_address() !=
      0) {
    return 1;
  }
  if (test_compute_escaped_write_handle_does_not_mutate_new_context_at_same_address() !=
      0) {
    return 1;
  }
  if (test_compute_escaped_write_target_does_not_mutate_new_context_at_same_address() !=
      0) {
    return 1;
  }
  return test_compute_escaped_access_does_not_mutate_new_context_at_same_address();
}

}  // namespace program_compute_contract
