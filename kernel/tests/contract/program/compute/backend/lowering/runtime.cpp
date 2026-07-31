#include "contract/program/compute/backend/lowering/local.hpp"
#include "test/assert.hpp"

#include <kernel/program/compute/binding/validation.hpp>

namespace program_compute_contract {
namespace {

using namespace backend_lowering_support;

int test_compute_runtime_bindings_validate_declared_obligations() {
  i32 input_a[4]{};
  i32 input_b[4]{};
  i32 output[4]{};
  i32 params[1]{7};
  rund::kernel::BufferSpan inputs[2]{
      rund::kernel::BufferSpan::contiguous(input_a, 4u),
      rund::kernel::BufferSpan::contiguous(input_b, 4u),
  };
  const rund::kernel::BindingObligations obligations{
      .tile_count = 4u,
      .input_buffer_count = 2u,
      .input_bytes_per_tile = 8u,
      .output_bytes_per_tile = 4u,
      .param_bytes = 4u,
  };
  rund::kernel::BindingSet bindings{
      .tile_count = 4u,
      .input_bytes_per_tile = 8u,
      .output_bytes_per_tile = 4u,
      .param_bytes = 4u,
      .input_buffers = inputs,
      .input_buffer_count = 2u,
      .param_data = params,
      .param_data_bytes = 4u,
      .staged_output = output,
      .staged_output_stride = 4u,
      .staged_output_count = 4u,
      .ok = true,
      .reason = "ok",
  };

  TEST_ASSERT(rund::kernel::ValidateRuntimeBindings(bindings, obligations).ok);

  rund::kernel::BindingObligations wrong_buffer_count = obligations;
  wrong_buffer_count.input_buffer_count = 3u;
  TEST_ASSERT(
      rund::kernel::ValidateRuntimeBindings(bindings, wrong_buffer_count)
          .reason == std::string_view{"compute_binding_input_count_mismatch"});

  rund::kernel::BindingSet null_input = bindings;
  inputs[0].data = nullptr;
  TEST_ASSERT(
      rund::kernel::ValidateRuntimeBindings(null_input, obligations).reason ==
      std::string_view{"compute_binding_input_null"});
  inputs[0] = rund::kernel::BufferSpan::contiguous(input_a, 4u);

  rund::kernel::BindingSet bad_stride = bindings;
  inputs[1].stride_bytes = 2u;
  TEST_ASSERT(
      rund::kernel::ValidateRuntimeBindings(bad_stride, obligations).reason ==
      std::string_view{"compute_binding_input_stride_invalid"});
  inputs[1] = rund::kernel::BufferSpan::contiguous(input_b, 4u);

  rund::kernel::BindingSet wrong_tile_count = bindings;
  wrong_tile_count.tile_count = 3u;
  TEST_ASSERT(
      rund::kernel::ValidateRuntimeBindings(wrong_tile_count, obligations)
          .reason == std::string_view{"compute_binding_tile_count_mismatch"});

  rund::kernel::BindingSet wrong_param_bytes = bindings;
  wrong_param_bytes.param_data_bytes = 3u;
  TEST_ASSERT(
      rund::kernel::ValidateRuntimeBindings(wrong_param_bytes, obligations)
          .reason == std::string_view{"compute_binding_param_size_mismatch"});

  rund::kernel::BindingSet null_param = bindings;
  null_param.param_data = nullptr;
  TEST_ASSERT(
      rund::kernel::ValidateRuntimeBindings(null_param, obligations).reason ==
      std::string_view{"compute_binding_param_null"});
  return 0;
}

} // namespace

int RunComputeBackendLoweringRuntimeContract() {
  return test_compute_runtime_bindings_validate_declared_obligations();
}

} // namespace program_compute_contract
