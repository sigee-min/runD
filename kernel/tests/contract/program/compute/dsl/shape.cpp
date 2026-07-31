#include "contract/program/compute/dsl/local.hpp"

namespace program_compute_contract {
namespace {

using namespace dsl_support;

int test_compute_public_values_cannot_be_forged_from_node_ids() {
  static_assert(!std::is_constructible_v<rund::compute_dsl::ComputeValue,
                                         rund::compute_dsl::detail::BuildContext*,
                                         rund::kernel::u32>);
  return 0;
}

int test_compute_public_handles_cannot_be_forged_from_binding_ids() {
  static_assert(!std::is_default_constructible_v<rund::compute_dsl::detail::ReadHandle>);
  static_assert(!std::is_default_constructible_v<rund::compute_dsl::detail::WriteHandle>);
  static_assert(!std::is_default_constructible_v<rund::compute_dsl::detail::WriteTarget>);
  static_assert(!std::is_constructible_v<rund::compute_dsl::detail::ReadHandle,
                                         rund::compute_dsl::detail::BuildContext*,
                                         rund::kernel::u32>);
  static_assert(!std::is_constructible_v<rund::compute_dsl::detail::WriteHandle,
                                         rund::compute_dsl::detail::BuildContext*,
                                         rund::kernel::u32>);
  static_assert(!std::is_constructible_v<rund::compute_dsl::detail::WriteTarget,
                                         rund::compute_dsl::detail::BuildContext*,
                                         rund::kernel::u32,
                                         bool>);
  return 0;
}

int test_compute_read_write_reject_non_buffer_objects() {
  using Body = decltype(rund::compute_dsl::bind(1u).fixed<1, 31>());
  using Array = i32[4];
  using ConstArray = const i32[4];
  using Vector = std::vector<i32>;

  static_assert(CanReadBuffer<Body, Array&>);
  static_assert(CanWriteBuffer<Body, Array&>);
  static_assert(CanReadBuffer<Body, ConstArray&>);
  static_assert(!CanWriteBuffer<Body, ConstArray&>);
  static_assert(CanReadBuffer<Body, i32*>);
  static_assert(CanWriteBuffer<Body, i32*>);
  static_assert(CanReadBuffer<Body, const i32*>);
  static_assert(!CanWriteBuffer<Body, const i32*>);
  static_assert(CanReadBuffer<Body, decltype(rund::compute_dsl::buffer<i32>())>);
  static_assert(CanWriteBuffer<Body, decltype(rund::compute_dsl::buffer<i32>())>);
  static_assert(CanReadBuffer<Body, decltype(rund::compute_dsl::buffer<const i32>())>);
  static_assert(
      !CanWriteBuffer<Body, decltype(rund::compute_dsl::buffer<const i32>())>);
  static_assert(!CanReadBuffer<Body, i32&>);
  static_assert(!CanWriteBuffer<Body, i32&>);
  static_assert(!CanReadBuffer<Body, Vector&>);
  static_assert(!CanWriteBuffer<Body, Vector&>);
  return 0;
}

}  // namespace

int RunComputeDslShapeContract() {
  if (test_compute_public_values_cannot_be_forged_from_node_ids() != 0) {
    return 1;
  }
  if (test_compute_public_handles_cannot_be_forged_from_binding_ids() != 0) {
    return 1;
  }
  return test_compute_read_write_reject_non_buffer_objects();
}

}  // namespace program_compute_contract
