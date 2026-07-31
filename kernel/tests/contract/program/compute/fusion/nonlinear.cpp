#include "test/assert.hpp"
#include "contract/program/compute/fusion/local/policy.hpp"

#include <kernel/program/compute/dsl.hpp>
#include <kernel/program/compute/fusion.hpp>
#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/lowering/artifact/admission.hpp>
#include <kernel/program/compute/lowering/entry.hpp>
#include <kernel/program/compute/lowering/fusion/build.hpp>

#include <array>

namespace program_compute_contract {
namespace {

using fusion_support::PolicyNode;

[[nodiscard]] rund::compute_dsl::ComputeOp BuildAddFiveOp() {
  std::array<rund::kernel::i32, 4u> input{};
  std::array<rund::kernel::i32, 4u> output{};
  const auto body =
      rund::compute_dsl::bind(input.size())
          .fixed<1, 31, rund::kernel::ComputeRounding::NearestEven,
                 rund::kernel::ComputeOverflow::Saturate,
                 rund::kernel::ComputeApproximation::Deterministic>()
          .param<"addend">(5)
          .read<"input">(input.data())
          .write<"middle">(output.data());
  return rund::compute_dsl::def("fusion-fixed-nonlinear-add-five")
      .on(body)
      .map([](auto i, auto b) {
        auto addend = b.template param<"addend">();
        auto input = b.template read<"input">();
        auto middle = b.template write<"middle">();
        middle[i] = input[i] + addend;
      });
}

[[nodiscard]] rund::compute_dsl::ComputeOp BuildNonlinearConsumerOp() {
  std::array<rund::kernel::i32, 4u> mode{};
  std::array<rund::kernel::i32, 4u> middle_input{};
  std::array<rund::kernel::i32, 4u> rhs{};
  std::array<rund::kernel::i32, 4u> output{};
  const auto body =
      rund::compute_dsl::bind(output.size())
          .fixed<1, 31, rund::kernel::ComputeRounding::NearestEven,
                 rund::kernel::ComputeOverflow::Saturate,
                 rund::kernel::ComputeApproximation::Deterministic>()
          .read<"mode">(mode.data())
          .read<"middle">(middle_input.data())
          .read<"rhs">(rhs.data())
          .write<"output">(output.data());
  return rund::compute_dsl::def("fusion-fixed-nonlinear-consumer")
      .on(body)
      .map([](auto i, auto b) {
        auto mode = b.template read<"mode">();
        auto middle = b.template read<"middle">();
        auto rhs = b.template read<"rhs">();
        auto output = b.template write<"output">();
        auto value = middle[i];
        auto result = rund::compute_dsl::rsqrt(value);
        result =
            rund::compute_dsl::select(rund::compute_dsl::eq(mode[i], 2),
                                      rund::compute_dsl::sqrt(value), result);
        result =
            rund::compute_dsl::select(rund::compute_dsl::eq(mode[i], 1),
                                      rund::compute_dsl::recip(value), result);
        output[i] = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], 0),
            rund::compute_dsl::div_fixed(value, rhs[i]), result);
      });
}

[[nodiscard]] rund::kernel::u32 FirstProducerOutputNode(
    const rund::kernel::compute_lowering_detail::ParsedIR &parsed) noexcept {
  rund::kernel::u32 add = 0u;
  for (std::size_t index = 0u; index < parsed.nodes.size(); ++index) {
    const auto &node = parsed.nodes[index];
    if (node.op == static_cast<rund::kernel::u8>(rund::kernel::IrOp::Add) &&
        node.aux == 0u) {
      add = static_cast<rund::kernel::u32>(index + 1u);
      break;
    }
  }
  for (std::size_t index = 0u; index < parsed.nodes.size(); ++index) {
    const auto &node = parsed.nodes[index];
    if (node.op ==
            static_cast<rund::kernel::u8>(rund::kernel::IrOp::Quantize) &&
        node.lhs == add) {
      return static_cast<rund::kernel::u32>(index + 1u);
    }
  }
  return 0u;
}

int test_compute_fusion_remaps_nonlinear_operands() {
  const rund::compute_dsl::ComputeOp first = BuildAddFiveOp();
  const rund::compute_dsl::ComputeOp second = BuildNonlinearConsumerOp();
  TEST_ASSERT(first.ok());
  TEST_ASSERT(second.ok());

  const rund::kernel::GraphBufferRef first_buffers[2] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::GraphBufferRef second_buffers[4] = {
      {.logical_id = 41u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 51u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 31u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::GraphNode nodes[2] = {
      {.op_hash_hi = first.ir().op_hash_hi,
       .op_hash_lo = first.ir().op_hash_lo,
       .buffers = first_buffers,
       .buffer_count = 2u,
       .element_count = 4u},
      {.op_hash_hi = second.ir().op_hash_hi,
       .op_hash_lo = second.ir().op_hash_lo,
       .buffers = second_buffers,
       .buffer_count = 4u,
       .element_count = 4u},
  };
  const rund::kernel::Graph graph{
      .nodes = nodes,
      .node_count = 2u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .domain = first.ir().domain,
      .fixed_format = first.ir().fixed_format,
  };
  const rund::kernel::FusionNodePolicy fusion_nodes[2] = {
      PolicyNode(first.ir()),
      PolicyNode(second.ir()),
  };
  const rund::kernel::FusionPolicy policy{
      .nodes = fusion_nodes,
      .node_count = 2u,
  };
  const rund::kernel::ComputeIR chain[2] = {first.ir(), second.ir()};
  const rund::kernel::ComputeFusedMapChainIR fused =
      rund::kernel::BuildFusedComputeMapChainIR(
          chain, 2u, graph, policy, rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(fused.ok);

  const rund::kernel::compute_lowering_detail::ParsedIR parsed =
      rund::kernel::compute_lowering_detail::ParseComputeIR(fused.ir);
  TEST_ASSERT(parsed.ok);
  const rund::kernel::u32 producer_node = FirstProducerOutputNode(parsed);
  TEST_ASSERT(producer_node != 0u);

  bool saw_div = false;
  bool saw_recip = false;
  bool saw_sqrt = false;
  bool saw_rsqrt = false;
  for (const auto &node : parsed.nodes) {
    const auto op = static_cast<rund::kernel::IrOp>(node.op);
    if (op == rund::kernel::IrOp::DivFixed) {
      TEST_ASSERT(node.lhs == producer_node);
      TEST_ASSERT(node.rhs != 0u);
      TEST_ASSERT(node.aux == 0u);
      saw_div = true;
    } else if (op == rund::kernel::IrOp::Recip ||
               op == rund::kernel::IrOp::Sqrt ||
               op == rund::kernel::IrOp::Rsqrt) {
      TEST_ASSERT(node.lhs == producer_node);
      TEST_ASSERT(node.rhs == 0u);
      TEST_ASSERT(node.aux == 0u);
      saw_recip = saw_recip || op == rund::kernel::IrOp::Recip;
      saw_sqrt = saw_sqrt || op == rund::kernel::IrOp::Sqrt;
      saw_rsqrt = saw_rsqrt || op == rund::kernel::IrOp::Rsqrt;
    }
  }
  TEST_ASSERT(saw_div);
  TEST_ASSERT(saw_recip);
  TEST_ASSERT(saw_sqrt);
  TEST_ASSERT(saw_rsqrt);

  const rund::kernel::LoweringArtifact artifact = rund::kernel::LowerComputeIR(
      fused.ir, rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(artifact.ok);
  return 0;
}

} // namespace

int RunComputeFixedNonlinearFusionContract() {
  return test_compute_fusion_remaps_nonlinear_operands();
}

} // namespace program_compute_contract
