#include "contract/program/compute/fixed/arithmetic/local.hpp"
#include "contract/program/compute/fusion/local/policy.hpp"

namespace program_compute_contract {
namespace {

using fusion_support::PolicyNode;

[[nodiscard]] rund::compute_dsl::ComputeOp
BuildFixedArithmeticMulAddConsumerOp() {
  std::array<rund::kernel::i32, 4u> input{};
  std::array<rund::kernel::i32, 4u> output{};
  const auto body = rund::compute_dsl::bind(input.size())
                        .fixed<1, 31>()
                        .read<"middle">(input.data())
                        .write<"output">(output.data());
  return rund::compute_dsl::def("fusion-fixed-arithmetic-mul-add-consumer")
      .on(body)
      .map([](auto i, auto b) {
        auto middle = b.template read<"middle">();
        auto output = b.template write<"output">();
        output[i] =
            rund::compute_dsl::mul_add_fixed(middle[i], middle[i], middle[i]);
      });
}

int test_compute_fixed_arithmetic_fusion_remaps_ternary_operands() {
  const rund::compute_dsl::ComputeOp first = [] {
    std::array<rund::kernel::i32, 4u> input{};
    std::array<rund::kernel::i32, 4u> output{};
    const auto body = rund::compute_dsl::bind(input.size())
                          .fixed<1, 31>()
                          .param<"addend">(5)
                          .read<"input">(input.data())
                          .write<"middle">(output.data());
    return rund::compute_dsl::def("fusion-fixed-arithmetic-add-five")
        .on(body)
        .map([](auto i, auto b) {
          auto addend = b.template param<"addend">();
          auto input = b.template read<"input">();
          auto middle = b.template write<"middle">();
          middle[i] = input[i] + addend;
        });
  }();
  const rund::compute_dsl::ComputeOp second =
      BuildFixedArithmeticMulAddConsumerOp();

  const rund::kernel::GraphBufferRef first_buffers[2] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::GraphBufferRef second_buffers[2] = {
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Read},
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
       .buffer_count = 2u,
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

  rund::kernel::u32 producer_node = 0u;
  for (std::size_t index = 0u; index < parsed.nodes.size(); ++index) {
    const auto &node = parsed.nodes[index];
    if (node.op == static_cast<rund::kernel::u8>(rund::kernel::IrOp::Add) &&
        node.aux == 0u) {
      producer_node = static_cast<rund::kernel::u32>(index + 1u);
      break;
    }
  }
  TEST_ASSERT(producer_node != 0u);
  for (std::size_t index = 0u; index < parsed.nodes.size(); ++index) {
    const auto &node = parsed.nodes[index];
    if (node.op ==
            static_cast<rund::kernel::u8>(rund::kernel::IrOp::Quantize) &&
        node.lhs == producer_node) {
      producer_node = static_cast<rund::kernel::u32>(index + 1u);
      break;
    }
  }
  TEST_ASSERT(parsed.nodes[producer_node - 1u].op ==
              static_cast<rund::kernel::u8>(rund::kernel::IrOp::Quantize));

  bool saw_mul_add = false;
  for (const auto &node : parsed.nodes) {
    if (node.op !=
        static_cast<rund::kernel::u8>(rund::kernel::IrOp::MulAddFixed)) {
      continue;
    }
    saw_mul_add = true;
    TEST_ASSERT(node.lhs == producer_node);
    TEST_ASSERT(node.rhs == producer_node);
    TEST_ASSERT(node.aux == producer_node);
  }
  TEST_ASSERT(saw_mul_add);
  return 0;
}

} // namespace

int RunComputeFixedArithmeticFusionContract() {
  return test_compute_fixed_arithmetic_fusion_remaps_ternary_operands();
}

} // namespace program_compute_contract
