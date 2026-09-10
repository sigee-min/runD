#include "contract/program/compute/dsl/local.hpp"

namespace program_compute_contract {
namespace {

using namespace dsl_support;

int test_compute_undeclared_capture_access_is_not_public_builder_shape() {
  const auto body = BuildIntegrateBody(7);

  static_assert(!decltype(body)::template has_read<"missing">());
  static_assert(!decltype(body)::template has_param<"missing">());
  static_assert(!decltype(body)::template has_write<"missing">());
  return 0;
}

} // namespace

int RunComputeDslIdentityContract() {
  if (RunComputeDslBasicIdentityContract() != 0) {
    return 1;
  }
  if (RunComputeDslFixedIdentityContract() != 0) {
    return 1;
  }
  return test_compute_undeclared_capture_access_is_not_public_builder_shape();
}

} // namespace program_compute_contract
