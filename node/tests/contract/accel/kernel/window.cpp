#include <accel/graph/factory/primitive/window.hpp>
#include <kernel/program/compute/window/reference.hpp>

#include <accel/api.hpp>
#include <accel/buffer.hpp>
#include <accel/check.hpp>
#include <accel/context/buffer.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/value.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>

#include <node/accel/context.hpp>
#include <node/accel/pick.hpp>

#include "../../../../src/accel/range_aggregate/plan.hpp"
#include "../../../../src/accel/window/shape.hpp"
#include "primitive/local.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace node_accel_contract {
namespace {

[[nodiscard]] constexpr rund::kernel::ComputeFixedFormat
FixedFormat(const rund::kernel::ComputeOverflow overflow) noexcept {
  return rund::kernel::ComputeFixedFormat{
      .integer_bits = 16u,
      .fraction_bits = 16u,
      .rounding = rund::kernel::ComputeRounding::NearestEven,
      .overflow = overflow,
      .approximation = rund::kernel::ComputeApproximation::Deterministic,
  };
}

[[nodiscard]] bool RunsWindow(const rund::AccelDevice &pick) {
  constexpr std::size_t kInputCount = 257u;
  constexpr std::size_t kOutputCount = 129u;
  std::array<rund::kernel::u32, kInputCount> input{};
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] = static_cast<rund::kernel::u32>((index * 37u + 11u) % 101u);
  }
  const rund::kernel::WindowDesc desc{
      .op = rund::kernel::WindowOp::Sum,
      .element = rund::kernel::WindowElement::U32,
      .boundary = rund::kernel::WindowBoundary::Clamp,
      .domain = rund::kernel::ComputeDomain::U32,
      .input_count = input.size(),
      .output_count = kOutputCount,
      .window_size = 129u,
      .stride = 2u,
      .pad_left = 64u,
  };
  const rund::kernel::WindowPlan plan = rund::kernel::PlanWindow(desc);
  std::array<rund::kernel::u32, kOutputCount> expected{};
  if (!rund::kernel::ReferenceWindowU32(input.data(), expected.data(), plan)
           .ok) {
    return false;
  }

  rund::AccelContext context = rund::node::accel::OpenAccel(pick);
  rund::AccelBuffer source = rund::node::accel::CreateAccelBuffer(
      context, rund::AccelBufferDesc{
                   .scalar_width_bytes = sizeof(input[0u]),
                   .count = input.size(),
                   .usage = rund::BufferUsage::ReadOnly,
               });
  rund::AccelBuffer output = rund::node::accel::CreateAccelBuffer(
      context, rund::AccelBufferDesc{
                   .scalar_width_bytes = sizeof(expected[0u]),
                   .count = expected.size(),
                   .usage = rund::BufferUsage::WriteOnly,
               });
  if (!context.check.ok || !source.check.ok || !output.check.ok ||
      !rund::node::accel::UploadAccelBuffer(context, source, input.data(),
                                            input.size() * sizeof(input[0u]))
           .ok) {
    return false;
  }
  const std::array<rund::AccelGraphBufferRef, 2u> refs{
      rund::AccelGraphBufferRef{.buffer = &source,
                                .role = rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{.buffer = &output,
                                .role = rund::kernel::BufferRole::Write}};
  const rund::AccelGraphNode node =
      rund::AccelWindow(refs.data(), refs.size(), desc);
  const rund::AccelKernel kernel = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = &node,
                   .node_count = 1u,
                   .scalar = rund::kernel::ComputeScalar::Lane32,
                   .domain = rund::kernel::ComputeDomain::U32,
               });
  if (!kernel.check.ok) {
    return false;
  }
  const std::array<rund::AccelRunBinding, 2u> bindings{
      rund::AccelRunBinding{.buffer = &source,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &output,
                            .role = rund::kernel::BufferRole::Write}};
  const rund::AccelEvidence evidence =
      rund::node::accel::RunAccelKernel(context, kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = kOutputCount,
                                            .fresh_evidence = true,
                                        });
  if (!evidence.outcome.ok || evidence.identity.backend != pick.api ||
      evidence.run.work.dispatch_count <= 1u ||
      evidence.run.work.original_dispatch_count <= 1u ||
      evidence.run.work.final_dispatch_count <= 1u) {
    return false;
  }
  std::array<rund::kernel::u32, kOutputCount> downloaded{};
  const rund::AccelCheck download = rund::node::accel::DownloadAccelBuffer(
      context, output, downloaded.data(),
      downloaded.size() * sizeof(downloaded[0u]));
  return download.ok && downloaded == expected;
}

[[nodiscard]] bool
RunsPaddedDirectWindow(const rund::AccelDevice &pick,
                       const rund::kernel::WindowBoundary boundary,
                       const std::array<rund::kernel::i32, 2u> &expected) {
  constexpr std::array<rund::kernel::i32, 3u> input{10, 20, 30};
  const rund::kernel::WindowDesc desc{
      .op = rund::kernel::WindowOp::Sum,
      .element = rund::kernel::WindowElement::U32,
      .boundary = boundary,
      .domain = rund::kernel::ComputeDomain::Fixed,
      .fixed_format = FixedFormat(rund::kernel::ComputeOverflow::Saturate),
      .input_count = input.size(),
      .output_count = expected.size(),
      .window_size = 5u,
      .stride = 5u,
      .pad_left = 4u,
  };
  const rund::kernel::WindowPlan plan = rund::kernel::PlanWindow(desc);
  std::array<rund::kernel::i32, 2u> reference{};
  if (!rund::kernel::ReferenceWindowFixedI32(input.data(), reference.data(),
                                             plan)
           .ok ||
      reference != expected) {
    return false;
  }

  rund::AccelContext context = rund::node::accel::OpenAccel(pick);
  rund::AccelBuffer source = rund::node::accel::CreateAccelBuffer(
      context, rund::AccelBufferDesc{
                   .scalar_width_bytes = sizeof(input[0u]),
                   .count = input.size(),
                   .usage = rund::BufferUsage::ReadOnly,
               });
  rund::AccelBuffer output = rund::node::accel::CreateAccelBuffer(
      context, rund::AccelBufferDesc{
                   .scalar_width_bytes = sizeof(expected[0u]),
                   .count = expected.size(),
                   .usage = rund::BufferUsage::WriteOnly,
               });
  if (!context.check.ok || !source.check.ok || !output.check.ok ||
      !rund::node::accel::UploadAccelBuffer(context, source, input.data(),
                                            input.size() * sizeof(input[0u]))
           .ok) {
    return false;
  }
  const std::array<rund::AccelGraphBufferRef, 2u> refs{
      rund::AccelGraphBufferRef{.buffer = &source,
                                .role = rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{.buffer = &output,
                                .role = rund::kernel::BufferRole::Write}};
  const rund::AccelGraphNode node =
      rund::AccelWindow(refs.data(), refs.size(), desc);
  const rund::AccelKernel kernel = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = &node,
                   .node_count = 1u,
                   .scalar = rund::kernel::ComputeScalar::Lane32,
                   .domain = rund::kernel::ComputeDomain::Fixed,
                   .fixed_format = desc.fixed_format,
               });
  if (!kernel.check.ok) {
    return false;
  }
  const std::array<rund::AccelRunBinding, 2u> bindings{
      rund::AccelRunBinding{.buffer = &source,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &output,
                            .role = rund::kernel::BufferRole::Write}};
  const rund::AccelEvidence evidence =
      rund::node::accel::RunAccelKernel(context, kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = expected.size(),
                                            .fresh_evidence = true,
                                        });
  if (!evidence.outcome.ok || evidence.identity.backend != pick.api ||
      evidence.run.work.dispatch_count != 1u ||
      evidence.run.work.original_dispatch_count != 1u ||
      evidence.run.work.final_dispatch_count != 1u) {
    return false;
  }
  std::array<rund::kernel::i32, 2u> downloaded{};
  const rund::AccelCheck download = rund::node::accel::DownloadAccelBuffer(
      context, output, downloaded.data(),
      downloaded.size() * sizeof(downloaded[0u]));
  return download.ok && downloaded == expected;
}

template <typename T, std::size_t InputCount = 257u,
          std::size_t WindowSize = 129u>
[[nodiscard]] bool RunsBlockWindow(const rund::AccelDevice &pick,
                                   const rund::kernel::WindowOp op,
                                   const rund::kernel::WindowElement element,
                                   const rund::kernel::ComputeDomain domain,
                                   const rund::kernel::WindowBoundary boundary =
                                       rund::kernel::WindowBoundary::Clip) {
  constexpr std::size_t kInputCount = InputCount;
  constexpr std::size_t kOutputCount = 65u;
  std::array<T, kInputCount> input{};
  for (std::size_t index = 0u; index < input.size(); ++index) {
    const auto value =
        static_cast<rund::kernel::i64>((index * 37u + 11u) % 211u);
    input[index] = static_cast<T>(std::is_signed_v<T> ? value - 105 : value);
  }
  if constexpr (sizeof(T) == 8u) {
    for (std::size_t index = 0u; index < input.size(); ++index) {
      input[index] = static_cast<T>(
          (static_cast<rund::kernel::u64>(index) * 0x9e3779b97f4a7c15ull) ^
          0xa5a5a5a55a5a5a5aull);
    }
  }
  input.front() = std::numeric_limits<T>::lowest();
  input.back() = std::numeric_limits<T>::max();
  const rund::kernel::WindowDesc desc{
      .op = op,
      .element = element,
      .boundary = boundary,
      .domain = domain,
      .input_count = input.size(),
      .output_count = kOutputCount,
      .window_size = WindowSize,
      .stride = 2u,
      .pad_left = WindowSize / 2u,
  };
  const rund::kernel::WindowPlan plan = rund::kernel::PlanWindow(desc);
  std::array<T, kOutputCount> expected{};
  const rund::kernel::WindowResult reference = [&]() {
    if constexpr (std::is_same_v<T, rund::kernel::i32>) {
      return rund::kernel::ReferenceWindowI32(input.data(), expected.data(),
                                              plan);
    } else if constexpr (std::is_same_v<T, rund::kernel::i64>) {
      return rund::kernel::ReferenceWindowI64(input.data(), expected.data(),
                                              plan);
    } else if constexpr (std::is_same_v<T, rund::kernel::u32>) {
      return rund::kernel::ReferenceWindowU32(input.data(), expected.data(),
                                              plan);
    } else {
      return rund::kernel::ReferenceWindowU64(input.data(), expected.data(),
                                              plan);
    }
  }();
  if (!reference.ok) {
    return false;
  }

  rund::AccelContext context = rund::node::accel::OpenAccel(pick);
  rund::AccelBuffer source = rund::node::accel::CreateAccelBuffer(
      context, rund::AccelBufferDesc{
                   .scalar_width_bytes = sizeof(T),
                   .count = input.size(),
                   .usage = rund::BufferUsage::ReadOnly,
               });
  rund::AccelBuffer output = rund::node::accel::CreateAccelBuffer(
      context, rund::AccelBufferDesc{
                   .scalar_width_bytes = sizeof(T),
                   .count = expected.size(),
                   .usage = rund::BufferUsage::WriteOnly,
               });
  if (!context.check.ok || !source.check.ok || !output.check.ok ||
      !rund::node::accel::UploadAccelBuffer(context, source, input.data(),
                                            input.size() * sizeof(T))
           .ok) {
    return false;
  }
  const std::array<rund::AccelGraphBufferRef, 2u> refs{
      rund::AccelGraphBufferRef{.buffer = &source,
                                .role = rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{.buffer = &output,
                                .role = rund::kernel::BufferRole::Write}};
  const rund::AccelGraphNode node =
      rund::AccelWindow(refs.data(), refs.size(), desc);
  const rund::AccelKernel kernel = rund::node::accel::CompileAccelKernel(
      context,
      rund::AccelGraph{
          .nodes = &node,
          .node_count = 1u,
          .scalar = sizeof(T) == 4u ? rund::kernel::ComputeScalar::Lane32
                                    : rund::kernel::ComputeScalar::Lane64,
          .domain = domain,
      });
  if (!kernel.check.ok) {
    return false;
  }
  const std::array<rund::AccelRunBinding, 2u> bindings{
      rund::AccelRunBinding{.buffer = &source,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &output,
                            .role = rund::kernel::BufferRole::Write}};
  const rund::AccelEvidence evidence =
      rund::node::accel::RunAccelKernel(context, kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = expected.size(),
                                            .fresh_evidence = true,
                                        });
  if (!evidence.outcome.ok || evidence.identity.backend != pick.api ||
      evidence.run.work.dispatch_count != 2u ||
      evidence.run.work.original_dispatch_count != 2u ||
      evidence.run.work.final_dispatch_count != 2u) {
    return false;
  }
  std::array<T, kOutputCount> downloaded{};
  const rund::AccelCheck download = rund::node::accel::DownloadAccelBuffer(
      context, output, downloaded.data(), downloaded.size() * sizeof(T));
  return download.ok && downloaded == expected;
}

[[nodiscard]] bool RequiredWindow(const rund::AccelApi api) {
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(primitive::Policy(api));
  constexpr std::array<rund::kernel::i32, 2u> clamp{50, 140};
  constexpr std::array<rund::kernel::i32, 2u> clip{10, 50};
  return pick.check.ok
             ? RunsWindow(pick) &&
                   RunsPaddedDirectWindow(
                       pick, rund::kernel::WindowBoundary::Clamp, clamp) &&
                   RunsPaddedDirectWindow(
                       pick, rund::kernel::WindowBoundary::Clip, clip) &&
                   RunsBlockWindow<rund::kernel::i32>(
                       pick, rund::kernel::WindowOp::Min,
                       rund::kernel::WindowElement::U32,
                       rund::kernel::ComputeDomain::I32) &&
                   RunsBlockWindow<rund::kernel::u64>(
                       pick, rund::kernel::WindowOp::Max,
                       rund::kernel::WindowElement::U64,
                       rund::kernel::ComputeDomain::U64) &&
                   RunsBlockWindow<rund::kernel::i64, 259u, 513u>(
                       pick, rund::kernel::WindowOp::Min,
                       rund::kernel::WindowElement::U64,
                       rund::kernel::ComputeDomain::I64,
                       rund::kernel::WindowBoundary::Clamp) &&
                   RunsBlockWindow<rund::kernel::u32, 259u, 513u>(
                       pick, rund::kernel::WindowOp::Max,
                       rund::kernel::WindowElement::U32,
                       rund::kernel::ComputeDomain::U32)
             : primitive::PickUnavailableReasonIsPrecise(pick, api);
}

} // namespace

bool WindowProjectionContract() {
  const rund::kernel::WindowDesc base{
      .op = rund::kernel::WindowOp::Sum,
      .element = rund::kernel::WindowElement::U32,
      .boundary = rund::kernel::WindowBoundary::Clip,
      .domain = rund::kernel::ComputeDomain::Fixed,
      .fixed_format = FixedFormat(rund::kernel::ComputeOverflow::Wrap),
      .input_count = 17u,
      .output_count = 9u,
      .window_size = 5u,
      .stride = 2u,
      .pad_left = 2u,
  };
  const rund::kernel::WindowPlan wrap_plan = rund::kernel::PlanWindow(base);
  const std::optional<rund::node::accel::detail::RangeShape> wrap_shape =
      rund::node::accel::detail::WindowRangeShape(wrap_plan);
  if (!wrap_plan.ok || !wrap_shape.has_value() ||
      wrap_shape->traits().arithmetic_law() !=
          rund::node::accel::detail::RangeLaw::ModuloWidth ||
      !wrap_shape->traits().invertible() || wrap_shape->input_count() != 17u ||
      wrap_shape->output_count() != 9u || wrap_shape->window_size() != 5u ||
      wrap_shape->stride() != 2u || wrap_shape->padding() != 2u ||
      wrap_shape->boundary() !=
          rund::node::accel::detail::RangeBoundary::Clip) {
    return false;
  }
  const rund::node::accel::detail::RangePlan cpu =
      rund::node::accel::detail::PlanRange(
          *wrap_shape, rund::node::accel::detail::RangeCaps::cpu_reference());
  if (!cpu.ok() ||
      cpu.candidate().disposition() !=
          rund::node::accel::detail::RangePath::Direct ||
      cpu.stage_count() != 1u || cpu.temporary_count() != 0u) {
    return false;
  }

  rund::kernel::WindowDesc saturating = base;
  saturating.fixed_format =
      FixedFormat(rund::kernel::ComputeOverflow::Saturate);
  const auto saturating_shape = rund::node::accel::detail::WindowRangeShape(
      rund::kernel::PlanWindow(saturating));
  if (!saturating_shape.has_value() ||
      saturating_shape->traits().arithmetic_law() !=
          rund::node::accel::detail::RangeLaw::Saturating ||
      saturating_shape->traits().invertible()) {
    return false;
  }

  rund::kernel::WindowDesc wider_than_input = base;
  wider_than_input.domain = rund::kernel::ComputeDomain::U32;
  wider_than_input.fixed_format = {};
  wider_than_input.input_count = 3u;
  wider_than_input.output_count = 1u;
  wider_than_input.window_size = 8u;
  wider_than_input.stride = 1u;
  wider_than_input.pad_left = 7u;
  const rund::kernel::WindowPlan wide_plan =
      rund::kernel::PlanWindow(wider_than_input);
  return wide_plan.ok &&
         rund::node::accel::detail::WindowRangeShape(wide_plan).has_value();
}

bool RequiredMetalRunsWindow() { return RequiredWindow(rund::AccelApi::Metal); }

bool RequiredVulkanRunsWindow() {
  return RequiredWindow(rund::AccelApi::Vulkan);
}

} // namespace node_accel_contract
