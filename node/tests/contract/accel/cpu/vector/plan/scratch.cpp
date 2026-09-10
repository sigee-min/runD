#include "layout.hpp"
#include "local.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string_view>
#include <vector>

namespace node_accel_contract {

[[nodiscard]] bool CheckIntegerScratch() {
  using namespace rund::node::accel::cpu_simd_detail;
  using namespace vector_plan;

  constexpr std::size_t kTileCount = 9u;
  constexpr rund::kernel::i32 kGap = -7777;
  const rund::kernel::CpuCaps caps = cpu::NeonCaps();

  // Non-Fixed execution never consumes the 128-bit materialization plane.
  // One read followed by x+x can destructively reuse its single dying slot;
  // the duplicate operand must not release that slot twice.
  std::array<rund::kernel::i32, kTileCount> integer_input{1,  -2, 3,  -4, 5,
                                                          -6, 7,  -8, 9};
  std::array<rund::kernel::i32, kTileCount> integer_output{};
  const auto integer_body = rund::compute_dsl::bind(kTileCount)
                                .i32()
                                .read<"input">(integer_input.data())
                                .write<"output">(integer_output.data());
  const rund::compute_dsl::ComputeOp integer_op =
      rund::compute_dsl::def("node-cpu-simd-integer-scratch")
          .on(integer_body)
          .map([](auto i, auto b) {
            const auto input = b.template read<"input">();
            const auto output = b.template write<"output">();
            const auto value = input[i];
            output[i] = value + value;
          });
  TEST_ASSERT(integer_op.ok());
  const rund::kernel::BindingSet integer_bindings =
      integer_op.bindings<rund::kernel::i32>(0u, caps.fixed_lane32_lanes,
                                             rund::kernel::ComputeApi::Cpu);
  const CpuSimdDispatch integer_dispatch =
      PrepareCpuSimdDispatch(integer_op.ir(), caps, integer_bindings);
  TEST_ASSERT(integer_dispatch.prepared.ok);
  const IndependentSlotLowerBound integer_lower_bound =
      IndependentPhysicalSlotLowerBound(integer_op.ir());
  TEST_ASSERT(integer_lower_bound.ok);
  TEST_ASSERT(integer_lower_bound.peak ==
              integer_dispatch.prepared.value_slot_count);
  TEST_ASSERT(integer_dispatch.prepared.value_slot_count == 1u);
  const std::size_t integer_raw_scratch =
      IndependentIntegerScratchBytes<rund::math32::simd::I32x>(1u);
  TEST_ASSERT(integer_dispatch.scratch_bytes(integer_dispatch.prepared) ==
              integer_raw_scratch);
  const std::size_t integer_scratch_words = ScratchWords(integer_raw_scratch);
  std::vector<std::max_align_t> integer_scratch(integer_scratch_words);
  CpuSimdBindingStorage integer_binding_storage{};
  const CpuSimdBindingView integer_binding_view =
      BindingView(integer_bindings, integer_binding_storage);
  const CpuSimdInvocation integer_invocation{
      .bindings = &integer_binding_view,
      .count = integer_bindings.tile_count,
  };
  const rund::node::accel::CpuSimdRunResult integer_run = integer_dispatch.run(
      integer_dispatch.prepared, integer_invocation,
      CpuSimdScratch{
          integer_scratch.data(),
          integer_scratch.size() * sizeof(std::max_align_t),
      });
  TEST_ASSERT(integer_run.ok);
  for (std::size_t index = 0u; index < kTileCount; ++index) {
    TEST_ASSERT(integer_output[index] == integer_input[index] * 2);
  }
  std::fill(integer_output.begin(), integer_output.end(), kGap);
  const rund::node::accel::CpuSimdRunResult integer_short =
      integer_dispatch.run(
          integer_dispatch.prepared, integer_invocation,
          CpuSimdScratch{
              integer_scratch.data(),
              (integer_scratch_words - 1u) * sizeof(std::max_align_t),
          });
  TEST_ASSERT(!integer_short.ok);
  TEST_ASSERT(std::string_view{integer_short.reason} ==
              "cpu_simd_scratch_invalid");
  TEST_ASSERT(std::all_of(integer_output.begin(), integer_output.end(),
                          [](const auto value) { return value == kGap; }));
  return true;
}

} // namespace node_accel_contract
