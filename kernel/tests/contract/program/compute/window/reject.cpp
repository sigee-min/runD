#include "local.hpp"

namespace program_compute_contract {
namespace {

int ExpectReason(const rund::kernel::WindowDesc &desc,
                 const std::string_view reason) {
  const rund::kernel::WindowPlan plan = rund::kernel::PlanWindow(desc);
  TEST_ASSERT(!plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == reason);
  return 0;
}

} // namespace

int WindowReject() {
  rund::kernel::WindowDesc desc = U32Window();
  desc.op = static_cast<rund::kernel::WindowOp>(0u);
  TEST_ASSERT(ExpectReason(desc, "compute_window_op_unsupported") == 0);

  desc = U32Window();
  desc.boundary = static_cast<rund::kernel::WindowBoundary>(0u);
  TEST_ASSERT(ExpectReason(desc, "compute_window_boundary_unsupported") == 0);

  desc = U32Window();
  desc.element = static_cast<rund::kernel::WindowElement>(0u);
  TEST_ASSERT(ExpectReason(desc, "compute_window_element_unsupported") == 0);

  desc = U32Window();
  desc.domain = rund::kernel::ComputeDomain::U64;
  TEST_ASSERT(ExpectReason(desc, "compute_window_domain_unsupported") == 0);

  desc = U32Window();
  desc.domain = static_cast<rund::kernel::ComputeDomain>(0u);
  TEST_ASSERT(ExpectReason(desc, "compute_window_domain_unsupported") == 0);

  desc = U32Window();
  desc.fixed_format = Fixed32(rund::kernel::ComputeOverflow::Wrap);
  TEST_ASSERT(ExpectReason(desc, "compute_window_fixed_unexpected") == 0);

  desc = U32Window();
  desc.domain = rund::kernel::ComputeDomain::Fixed;
  desc.fixed_format = Fixed32(rund::kernel::ComputeOverflow::Wrap);
  desc.fixed_format.integer_bits = 15u;
  TEST_ASSERT(ExpectReason(desc, "compute_window_fixed_invalid") == 0);

  desc = U32Window();
  desc.window_size = 0u;
  TEST_ASSERT(ExpectReason(desc, "compute_window_size_invalid") == 0);

  desc = U32Window();
  desc.stride = 0u;
  TEST_ASSERT(ExpectReason(desc, "compute_window_stride_invalid") == 0);

  desc = U32Window();
  desc.pad_left = desc.window_size;
  TEST_ASSERT(ExpectReason(desc, "compute_window_padding_invalid") == 0);

  desc = U32Window();
  desc.input_count = 0u;
  TEST_ASSERT(ExpectReason(desc, "compute_window_count_invalid") == 0);

  desc = U32Window();
  desc.output_count = 0u;
  TEST_ASSERT(ExpectReason(desc, "compute_window_count_invalid") == 0);

  desc = U32Window();
  desc.input_count = 0u;
  desc.output_count = 0u;
  desc.op = rund::kernel::WindowOp::Min;
  TEST_ASSERT(ExpectReason(desc, "compute_window_count_zero") == 0);

  desc = U32Window();
  desc.input_count = 4u;
  desc.output_count = 3u;
  desc.window_size = 2u;
  desc.stride = 3u;
  desc.pad_left = 0u;
  TEST_ASSERT(ExpectReason(desc, "compute_window_shape_invalid") == 0);

  desc = U32Window();
  desc.input_count = std::numeric_limits<rund::kernel::u64>::max();
  desc.output_count = std::numeric_limits<rund::kernel::u64>::max();
  desc.window_size = 1u;
  desc.stride = 2u;
  desc.pad_left = 0u;
  TEST_ASSERT(ExpectReason(desc, "compute_window_shape_invalid") == 0);

  desc = U32Window();
  desc.element = rund::kernel::WindowElement::U64;
  desc.domain = rund::kernel::ComputeDomain::U64;
  desc.input_count = std::numeric_limits<rund::kernel::u64>::max() / 8u + 1u;
  desc.output_count = 1u;
  desc.window_size = 1u;
  desc.stride = 1u;
  desc.pad_left = 0u;
  TEST_ASSERT(ExpectReason(desc, "compute_window_bytes_overflow") == 0);
  return 0;
}

} // namespace program_compute_contract
