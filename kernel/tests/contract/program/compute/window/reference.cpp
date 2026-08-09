#include "local.hpp"

namespace program_compute_contract {

int WindowReference() {
  const std::array<rund::kernel::u32, 6u> input{1u, 2u, 3u, 4u, 5u, 6u};
  std::array<rund::kernel::u32, 4u> clamp{};
  const rund::kernel::WindowPlan clamp_plan =
      rund::kernel::PlanWindow(U32Window());
  const rund::kernel::WindowResult clamp_result =
      rund::kernel::ReferenceWindowU32(input.data(), clamp.data(), clamp_plan);
  TEST_ASSERT(clamp_result.ok);
  TEST_ASSERT(clamp == (std::array<rund::kernel::u32, 4u>{4u, 9u, 15u, 18u}));

  rund::kernel::WindowDesc clip_desc = U32Window();
  clip_desc.boundary = rund::kernel::WindowBoundary::Clip;
  const rund::kernel::WindowPlan clip_plan =
      rund::kernel::PlanWindow(clip_desc);
  std::array<rund::kernel::u32, 4u> clip{};
  const rund::kernel::WindowResult clip_result =
      rund::kernel::ReferenceWindowU32(input.data(), clip.data(), clip_plan);
  TEST_ASSERT(clip_result.ok);
  TEST_ASSERT(clip == (std::array<rund::kernel::u32, 4u>{3u, 9u, 15u, 6u}));

  const std::array<rund::kernel::u32, 3u> padded_input{10u, 20u, 30u};
  rund::kernel::WindowDesc padded_desc{
      .op = rund::kernel::WindowOp::Sum,
      .element = rund::kernel::WindowElement::U32,
      .boundary = rund::kernel::WindowBoundary::Clamp,
      .domain = rund::kernel::ComputeDomain::U32,
      .input_count = padded_input.size(),
      .output_count = 2u,
      .window_size = 5u,
      .stride = 5u,
      .pad_left = 4u,
  };
  std::array<rund::kernel::u32, 2u> padded_output{};
  TEST_ASSERT(rund::kernel::ReferenceWindowU32(
                  padded_input.data(), padded_output.data(),
                  rund::kernel::PlanWindow(padded_desc))
                  .ok);
  TEST_ASSERT(padded_output == (std::array<rund::kernel::u32, 2u>{50u, 140u}));
  padded_desc.boundary = rund::kernel::WindowBoundary::Clip;
  TEST_ASSERT(rund::kernel::ReferenceWindowU32(
                  padded_input.data(), padded_output.data(),
                  rund::kernel::PlanWindow(padded_desc))
                  .ok);
  TEST_ASSERT(padded_output == (std::array<rund::kernel::u32, 2u>{10u, 50u}));

  const std::array<rund::kernel::i32, 5u> signed_input{-5, 2, -9, 7, 1};
  rund::kernel::WindowDesc extrema_desc{
      .op = rund::kernel::WindowOp::Min,
      .element = rund::kernel::WindowElement::U32,
      .boundary = rund::kernel::WindowBoundary::Clip,
      .domain = rund::kernel::ComputeDomain::I32,
      .input_count = 5u,
      .output_count = 3u,
      .window_size = 3u,
      .stride = 2u,
      .pad_left = 1u,
  };
  std::array<rund::kernel::i32, 3u> minimum{};
  TEST_ASSERT(
      rund::kernel::ReferenceWindowI32(signed_input.data(), minimum.data(),
                                       rund::kernel::PlanWindow(extrema_desc))
          .ok);
  TEST_ASSERT(minimum == (std::array<rund::kernel::i32, 3u>{-5, -9, 1}));
  extrema_desc.op = rund::kernel::WindowOp::Max;
  std::array<rund::kernel::i32, 3u> maximum{};
  TEST_ASSERT(
      rund::kernel::ReferenceWindowI32(signed_input.data(), maximum.data(),
                                       rund::kernel::PlanWindow(extrema_desc))
          .ok);
  TEST_ASSERT(maximum == (std::array<rund::kernel::i32, 3u>{2, 7, 7}));

  rund::kernel::WindowDesc fixed_desc{
      .op = rund::kernel::WindowOp::Sum,
      .element = rund::kernel::WindowElement::U32,
      .boundary = rund::kernel::WindowBoundary::Clip,
      .domain = rund::kernel::ComputeDomain::Fixed,
      .fixed_format = Fixed32(rund::kernel::ComputeOverflow::Saturate),
      .input_count = 2u,
      .output_count = 1u,
      .window_size = 2u,
      .stride = 1u,
      .pad_left = 0u,
  };
  const std::array<rund::kernel::i32, 2u> fixed_input{
      std::numeric_limits<rund::kernel::i32>::max() - 1, 10};
  std::array<rund::kernel::i32, 1u> fixed_output{};
  TEST_ASSERT(rund::kernel::ReferenceWindowFixedI32(
                  fixed_input.data(), fixed_output.data(),
                  rund::kernel::PlanWindow(fixed_desc))
                  .ok);
  TEST_ASSERT(fixed_output[0] == std::numeric_limits<rund::kernel::i32>::max());

  fixed_desc.fixed_format.overflow = rund::kernel::ComputeOverflow::Wrap;
  TEST_ASSERT(rund::kernel::ReferenceWindowFixedI32(
                  fixed_input.data(), fixed_output.data(),
                  rund::kernel::PlanWindow(fixed_desc))
                  .ok);
  TEST_ASSERT(fixed_output[0] == std::bit_cast<rund::kernel::i32>(0x80000008u));

  rund::kernel::WindowDesc wrap_desc = U32Window();
  wrap_desc.input_count = 2u;
  wrap_desc.output_count = 1u;
  wrap_desc.window_size = 2u;
  wrap_desc.stride = 1u;
  wrap_desc.pad_left = 0u;
  const std::array<rund::kernel::u32, 2u> wrap_input{
      std::numeric_limits<rund::kernel::u32>::max(), 2u};
  std::array<rund::kernel::u32, 1u> wrap_output{};
  TEST_ASSERT(
      rund::kernel::ReferenceWindowU32(wrap_input.data(), wrap_output.data(),
                                       rund::kernel::PlanWindow(wrap_desc))
          .ok);
  TEST_ASSERT(wrap_output[0] == 1u);

  rund::kernel::WindowDesc zero_desc = U32Window();
  zero_desc.input_count = 0u;
  zero_desc.output_count = 0u;
  TEST_ASSERT(rund::kernel::ReferenceWindowU32(
                  nullptr, nullptr, rund::kernel::PlanWindow(zero_desc))
                  .ok);
  const rund::kernel::WindowResult missing =
      rund::kernel::ReferenceWindowU32(nullptr, nullptr, clamp_plan);
  TEST_ASSERT(!missing.ok);
  TEST_ASSERT(std::string_view{missing.reason} ==
              "compute_window_buffer_invalid");
  return 0;
}

} // namespace program_compute_contract
