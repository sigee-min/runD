#pragma once

#include "layout.hpp"
#include "local.hpp"

#include <bit>
#include <type_traits>
#include <vector>

namespace node_accel_contract {

template <class S>
[[nodiscard]] bool CheckPreparedAffine(const std::size_t stride,
                                       const bool in_place) {
  using namespace rund::node::accel::cpu_simd_detail;
  using U = std::make_unsigned_t<S>;
  constexpr std::size_t count = 37u;
  std::array<S, count * 2u> input{}, output{};
  std::array<U, count> original{};
  for (std::size_t i = 0u; i < count; ++i) {
    original[i] = static_cast<U>(~U{0u} - static_cast<U>(i * 2654435761ull));
    input[i * stride] = std::bit_cast<S>(original[i]);
  }
  const auto mode = [] {
    if constexpr (sizeof(S) == 4u)
      return rund::compute_dsl::bind(count).i32();
    else
      return rund::compute_dsl::bind(count).i64();
  }();
  const auto body =
      mode.template param<"factor">(S{3})
          .template read<"input">(input.data())
          .template write<"output">(in_place ? input.data() : output.data());
  const auto op = rund::compute_dsl::def("cpu-prepared-affine")
                      .on(body)
                      .map([](auto i, auto b) {
                        static_cast<void>(b.index());
    const auto x = b.template read<"input">()[i];
                        const auto p = b.template param<"factor">();
                        b.template write<"output">()[i] =
                            ((x * p + 7) * 5 - 11) * 9 + 13;
                      });
  TEST_ASSERT(op.ok());
  const auto caps = cpu::NeonCaps();
  const auto lanes =
      sizeof(S) == 4u ? caps.fixed_lane32_lanes : caps.fixed_lane64_lanes;
  auto bindings =
      op.template bindings<S>(0u, lanes, rund::kernel::ComputeApi::Cpu);
  std::array<rund::kernel::BufferSpan, 1u> reads{bindings.input_buffers[0u]};
  std::array<rund::kernel::OutputSpan, 1u> writes{bindings.output_buffers[0u]};
  reads[0u].stride_bytes = stride * sizeof(S);
  writes[0u].stride_bytes = stride * sizeof(S);
  bindings.input_buffers = reads.data();
  bindings.output_buffers = writes.data();
  auto dispatch = PrepareCpuSimdDispatch(op.ir(), caps, bindings);
  TEST_ASSERT(dispatch.prepared.ok && dispatch.prepared.affine.valid);
  TEST_ASSERT(static_cast<U>(dispatch.prepared.affine.multiplier) == U{135});
  TEST_ASSERT(static_cast<U>(dispatch.prepared.affine.addend) == U{229});
  std::vector<std::max_align_t> scratch(
      vector_plan::ScratchWords(dispatch.scratch_bytes(dispatch.prepared)));
  CpuSimdBindingStorage storage{};
  const auto view = BindingView(bindings, storage);
  // Non-zero begins, full vectors and tails; unchanged prefix/suffix canaries.
  for (const bool optimized : {true, false}) {
    dispatch.prepared.affine.valid = optimized;
    for (std::size_t i = 0; i < count; ++i)
      input[i * stride] = std::bit_cast<S>(original[i]);
    output.fill(S{});
    const auto result = dispatch.run(
        dispatch.prepared,
        CpuSimdInvocation{.bindings = &view, .begin = 1u, .count = count - 2u},
        CpuSimdScratch{scratch.data(),
                       scratch.size() * sizeof(std::max_align_t)});
    TEST_ASSERT(result.ok && result.processed_tiles == count - 2u);
    TEST_ASSERT(result.vector_chunk_count == (count - 2u) / lanes);
    TEST_ASSERT(result.tail_chunk_count ==
                ((count - 2u) % lanes != 0u ? 1u : 0u));
    const auto &target = in_place ? input : output;
    for (std::size_t i = 0; i < count; ++i) {
      const U expected =
          i == 0u || i == count - 1u
              ? (in_place ? original[i] : U{})
              : static_cast<U>(((original[i] * U{3} + U{7}) * U{5} - U{11}) *
                                   U{9} +
                               U{13});
      TEST_ASSERT(std::bit_cast<U>(target[i * stride]) == expected);
      if (stride == 2u)
        TEST_ASSERT(target[i * stride + 1u] == S{});
    }
  }
  return true;
}

} // namespace node_accel_contract
