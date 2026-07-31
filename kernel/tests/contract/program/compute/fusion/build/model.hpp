#pragma once

#include "contract/program/compute/fusion/local.hpp"
#include "test/assert.hpp"

#include <kernel/program/compute/lowering/emission.hpp>

#include <limits>
#include <string>
#include <string_view>

namespace program_compute_contract::fusion_build_contract {

using namespace fusion_support;

inline constexpr rund::kernel::ComputeFixedFormat kCarrierFormat16x16{
    .integer_bits = 16u,
    .fraction_bits = 16u,
    .rounding = rund::kernel::ComputeRounding::NearestEven,
    .overflow = rund::kernel::ComputeOverflow::Saturate,
    .approximation = rund::kernel::ComputeApproximation::Exact,
};

template <rund::compute_dsl::detail::ScalarMode HeaderMode,
          rund::compute_dsl::detail::ScalarMode OutputMode = HeaderMode>
struct CarrierBody final {
  [[nodiscard]] static constexpr rund::compute_dsl::detail::ScalarMode
  scalar_mode() noexcept {
    return HeaderMode;
  }
  [[nodiscard]] const std::vector<rund::compute_dsl::detail::BindingRuntime> &
  bindings() const noexcept {
    return values;
  }
  [[nodiscard]] constexpr rund::kernel::u64 tile_count() const noexcept {
    return 4u;
  }
  [[nodiscard]] constexpr rund::kernel::ComputeFixedFormat
  fixed_format() const noexcept {
    return {};
  }
  [[nodiscard]] constexpr bool ok() const noexcept { return true; }
  [[nodiscard]] constexpr const char *reason() const noexcept { return "ok"; }

  std::vector<rund::compute_dsl::detail::BindingRuntime> values{
      rund::compute_dsl::detail::BindingRuntime{
          .kind = rund::compute_dsl::detail::BindingKind::Read,
          .numeric_mode = HeaderMode,
          .name = "input",
          .element_bytes =
              rund::compute_dsl::detail::WideMode(HeaderMode) ? 8u : 4u,
      },
      rund::compute_dsl::detail::BindingRuntime{
          .kind = rund::compute_dsl::detail::BindingKind::Write,
          .numeric_mode = OutputMode,
          .name = "output",
          .element_bytes =
              rund::compute_dsl::detail::WideMode(HeaderMode) ? 8u : 4u,
      },
  };
};

template <rund::compute_dsl::detail::ScalarMode Mode>
[[nodiscard]] rund::compute_dsl::ComputeOp BuildCarrierProducerOp() {
  using namespace rund::compute_dsl::detail;
  CarrierBody<Mode> body{};
  BuildContext context{body.bindings(), Mode};
  const auto input = DynamicRead(context, 0u);
  const auto zero = TypedConstant(input, Mode, 0u);
  DynamicWrite(context, 1u, Binary(rund::kernel::IrOp::Add, input, zero));
  rund::kernel::ComputeIR ir =
      BuildIr("fusion-carrier-producer", body, context);
  const rund::kernel::ComputeMap map = BuildMap(ir, body);
  return rund::compute_dsl::ComputeOp{std::move(ir), map,
                                      std::move(body.values), 4u};
}

template <rund::kernel::IrWriteMode WriteMode>
[[nodiscard]] rund::compute_dsl::ComputeOp BuildCarrierConsumerOp() {
  using namespace rund::compute_dsl::detail;
  constexpr ScalarMode source_mode = ScalarMode::U32;
  constexpr ScalarMode target_mode =
      WriteMode == rund::kernel::IrWriteMode::CheckedOrdinal
          ? ScalarMode::I32
          : ScalarMode::FixedLane32;
  CarrierBody<source_mode, target_mode> body{};
  BuildContext context{body.bindings(), source_mode};
  const auto input = DynamicRead(context, 0u);
  const auto zero = TypedConstant(input, source_mode, 0u);
  if constexpr (WriteMode == rund::kernel::IrWriteMode::CheckedOrdinal) {
    const auto limit =
        TypedConstant(input, source_mode,
                      static_cast<rund::kernel::u64>(
                          std::numeric_limits<rund::kernel::i32>::max()));
    const auto representable =
        Binary(rund::kernel::IrOp::LeUnsigned, input, limit);
    DynamicCheckedOrdinalWrite(
        context, 1u,
        Ternary(rund::kernel::IrOp::Select, representable, input, zero));
  } else {
    const auto predicate = Binary(rund::kernel::IrOp::Ne, input, zero);
    const auto one = TypedConstant(input, source_mode, 1u);
    DynamicBoundaryMaskWrite(
        context, 1u, Ternary(rund::kernel::IrOp::Select, predicate, one, zero),
        kCarrierFormat16x16);
  }
  rund::kernel::ComputeIR ir =
      BuildIr(WriteMode == rund::kernel::IrWriteMode::CheckedOrdinal
                  ? "fusion-checked-ordinal"
                  : "fusion-boundary-mask",
              body, context);
  const rund::kernel::ComputeMap map = BuildMap(ir, body);
  return rund::compute_dsl::ComputeOp{std::move(ir), map,
                                      std::move(body.values), 4u};
}

struct Pair final {
  Pair(const rund::compute_dsl::ComputeOp &first,
       const rund::compute_dsl::ComputeOp &second)
      : nodes{{.op_hash_hi = first.ir().op_hash_hi,
               .op_hash_lo = first.ir().op_hash_lo,
               .buffers = first_buffers,
               .buffer_count = 2u,
               .element_count = 4u},
              {.op_hash_hi = second.ir().op_hash_hi,
               .op_hash_lo = second.ir().op_hash_lo,
               .buffers = second_buffers,
               .buffer_count = 2u,
               .element_count = 4u}},
        fusion_nodes{PolicyNode(first.ir()), PolicyNode(second.ir())},
        chain{first.ir(), second.ir()},
        graph{.nodes = nodes,
              .node_count = 2u,
              .scalar = rund::kernel::ComputeScalar::Lane32,
              .domain = first.ir().domain,
              .fixed_format = first.ir().fixed_format},
        policy{.nodes = fusion_nodes, .node_count = 2u} {}

  Pair(const Pair &) = delete;
  Pair &operator=(const Pair &) = delete;
  Pair(Pair &&) = delete;
  Pair &operator=(Pair &&) = delete;

  rund::kernel::GraphBufferRef first_buffers[2] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
  };
  rund::kernel::GraphBufferRef second_buffers[2] = {
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 31u, .role = rund::kernel::BufferRole::Write},
  };
  rund::kernel::GraphNode nodes[2]{};
  rund::kernel::FusionNodePolicy fusion_nodes[2]{};
  rund::kernel::ComputeIR chain[2]{};
  rund::kernel::Graph graph{};
  rund::kernel::FusionPolicy policy{};
};

[[nodiscard]] inline rund::kernel::ComputeFusedMapChainIR
BuildPair(const rund::compute_dsl::ComputeOp &first,
          const rund::compute_dsl::ComputeOp &second,
          const rund::kernel::ComputeApi api) {
  const Pair pair{first, second};
  return rund::kernel::BuildFusedComputeMapChainIR(pair.chain, 2u, pair.graph,
                                                   pair.policy, api);
}

} // namespace program_compute_contract::fusion_build_contract
