#include <accel/api.hpp>
#include <accel/buffer.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>
#include <accel/graph/value.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>

#include <accel/graph/factory/primitive/matrix/node.hpp>

#include "primitive/local.hpp"
#include "test/compute/fixed.hpp"
#include <node/accel/context.hpp>
#include <node/accel/pick.hpp>
#if defined(__APPLE__)
#include "src/accel/metal/numeric/source.hpp"
#endif
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include "src/accel/vulkan/numeric/source.hpp"
#endif

#include <array>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <type_traits>

namespace node_accel_contract {
namespace {

[[nodiscard]] bool MatrixFailure(const rund::AccelDevice &pick,
                                 const char *const phase,
                                 const char *const reason) {
  std::fprintf(stderr, "matrix failure: backend=%u phase=%s reason=%s\n",
               static_cast<unsigned>(pick.api), phase, reason);
  return false;
}

#if defined(__APPLE__) || defined(RUND_NODE_HAVE_VULKAN_SDK)
[[nodiscard]] bool OrderedToken(const std::string &source,
                                const std::string_view earlier,
                                const std::string_view later) {
  const std::size_t earlier_offset = source.find(earlier);
  const std::size_t later_offset = source.find(later);
  return earlier_offset != std::string::npos &&
         later_offset != std::string::npos && earlier_offset < later_offset;
}
#endif

[[nodiscard]] bool NumericGeneratedSourcesCloseGenericDependencies() {
#if defined(__APPLE__)
  const std::string metal =
      rund::node::accel::detail::MetalNumericFixedLane64Source();
  if (!OrderedToken(metal, "inline ulong RundUnsignedDivU128ByU64(",
                    "ulong magnitude = RundUnsignedDivU128ByU64(") ||
      !OrderedToken(metal, "inline ulong RundUnsignedSqrtU128ToU64(",
                    "ulong root = RundUnsignedSqrtU128ToU64(")) {
    return false;
  }
#endif
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const std::string vulkan = rund::node::accel::detail::NumericBaseSource64();
  if (!OrderedToken(vulkan, "uint64_t RundUnsignedDivU128ByU64(",
                    "uint64_t magnitude = RundUnsignedDivU128ByU64(") ||
      !OrderedToken(vulkan, "uint64_t RundUnsignedSqrtU128ToU64(",
                    "uint64_t root = RundUnsignedSqrtU128ToU64(")) {
    return false;
  }
#endif
  return true;
}

template <typename Value, std::size_t LeftCount, std::size_t RightCount,
          std::size_t OutputCount>
[[nodiscard]] bool RunMatrix(const rund::AccelDevice &pick,
                             const rund::kernel::MatrixOp op,
                             const rund::kernel::MatrixShape shape,
                             const std::array<Value, LeftCount> &left,
                             const std::array<Value, RightCount> &right,
                             const std::array<Value, OutputCount> &expected) {
  namespace fix = node_accel_contract::primitive;
  const rund::kernel::ComputeScalar scalar =
      sizeof(Value) == sizeof(rund::kernel::i64)
          ? rund::kernel::ComputeScalar::Lane64
          : rund::kernel::ComputeScalar::Lane32;
  auto context = rund::node::accel::OpenAccel(pick);
  if (!context.check.ok) {
    return MatrixFailure(pick, "open", context.check.reason);
  }
  auto left_buffer = rund::node::accel::CreateAccelBuffer(
      context,
      fix::BufferDesc(rund::BufferUsage::ReadOnly, sizeof(Value), left.size()));
  auto right_buffer = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::ReadOnly, sizeof(Value),
                               right.size()));
  auto output_buffer = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::WriteOnly, sizeof(Value),
                               expected.size()));
  if (!left_buffer.check.ok || !output_buffer.check.ok ||
      !rund::node::accel::UploadAccelBuffer(context, left_buffer, left.data(),
                                            left.size() * sizeof(Value))
           .ok) {
    return MatrixFailure(pick, "input", "buffer_or_upload");
  }
  if (op != rund::kernel::MatrixOp::Transpose &&
      (!right_buffer.check.ok ||
       !rund::node::accel::UploadAccelBuffer(
            context, right_buffer, right.data(), right.size() * sizeof(Value))
            .ok)) {
    return MatrixFailure(pick, "right", "buffer_or_upload");
  }

  const std::array<rund::AccelGraphBufferRef, 3u> refs{
      rund::AccelGraphBufferRef{.buffer = &left_buffer,
                                .role = rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{
          .buffer = op == rund::kernel::MatrixOp::Transpose ? &output_buffer
                                                            : &right_buffer,
          .role = op == rund::kernel::MatrixOp::Transpose
                      ? rund::kernel::BufferRole::Write
                      : rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{.buffer = &output_buffer,
                                .role = rund::kernel::BufferRole::Write},
  };
  const std::uint64_t ref_count =
      op == rund::kernel::MatrixOp::Transpose ? 2u : 3u;
  const rund::kernel::MatrixDesc desc{
      .op = op,
      .layout = shape.layout,
      .arithmetic = shape.arithmetic,
      .rows = shape.rows,
      .cols = shape.cols,
      .inner = shape.inner,
      .batch_count = shape.batch_count,
      .element_bytes = shape.element_bytes,
      .fixed_format =
          shape.arithmetic == rund::kernel::MatrixArithmetic::Fixed
              ? test::FixedFormatForLane(
                    scalar, rund::kernel::ComputeApproximation::Deterministic)
              : rund::kernel::ComputeFixedFormat{},
  };
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelMatrix(refs.data(), ref_count, desc),
  };
  const auto kernel = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = nodes.data(),
                   .node_count = nodes.size(),
                   .scalar = scalar,
                   .domain = rund::kernel::ComputeDomain::Fixed,
                   .fixed_format = test::FixedFormatForLane(scalar),
               });
  if (!kernel.check.ok) {
    return MatrixFailure(pick, "compile", kernel.check.reason);
  }
  const std::array<rund::AccelRunBinding, 3u> bindings{
      rund::AccelRunBinding{.buffer = &left_buffer,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = op == rund::kernel::MatrixOp::Transpose
                                          ? &output_buffer
                                          : &right_buffer,
                            .role = op == rund::kernel::MatrixOp::Transpose
                                        ? rund::kernel::BufferRole::Write
                                        : rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &output_buffer,
                            .role = rund::kernel::BufferRole::Write},
  };
  const auto evidence =
      rund::node::accel::RunAccelKernel(context, kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = ref_count,
                                            .tile_count = expected.size(),
                                            .fresh_evidence = true,
                                        });
  if (!evidence.ok || evidence.dispatch_count != 1u) {
    return MatrixFailure(pick, "run", evidence.reason);
  }
  std::array<Value, OutputCount> downloaded{};
  const rund::AccelCheck download = rund::node::accel::DownloadAccelBuffer(
      context, output_buffer, downloaded.data(),
      downloaded.size() * sizeof(Value));
  if (!download.ok) {
    return MatrixFailure(pick, "download", download.reason);
  }
  const std::uint64_t actual =
      fix::HashValues(downloaded.data(), downloaded.size());
  const std::uint64_t wanted =
      fix::HashValues(expected.data(), expected.size());
  if (actual != wanted) {
    std::fprintf(stderr,
                 "matrix failure: backend=%u phase=compare actual=%llu "
                 "expected=%llu rows=%llu cols=%llu inner=%llu width=%zu\n",
                 static_cast<unsigned>(pick.api),
                 static_cast<unsigned long long>(actual),
                 static_cast<unsigned long long>(wanted),
                 static_cast<unsigned long long>(shape.rows),
                 static_cast<unsigned long long>(shape.cols),
                 static_cast<unsigned long long>(shape.inner), sizeof(Value));
    return false;
  }
  return true;
}

template <typename Value>
[[nodiscard]] bool RunTiledMatrix(const rund::AccelDevice &pick) {
  constexpr std::size_t rows = 17u;
  constexpr std::size_t cols = 18u;
  constexpr std::size_t inner = 17u;
  constexpr unsigned fraction_bits = sizeof(Value) * 8u - 1u;
  using Wide = std::conditional_t<sizeof(Value) == sizeof(std::int32_t),
                                  std::int64_t, __int128_t>;
  constexpr Value base = static_cast<Value>(Value{1} << (fraction_bits - 6u));
  std::array<Value, rows * inner> left{};
  std::array<Value, inner * cols> right{};
  std::array<Value, rows * cols> expected{};
  for (std::size_t row = 0u; row < rows; ++row) {
    for (std::size_t k = 0u; k < inner; ++k) {
      left[k * rows + row] = static_cast<Value>(
          base * static_cast<Value>(1u + (row + 2u * k) % 7u));
    }
  }
  for (std::size_t col = 0u; col < cols; ++col) {
    for (std::size_t k = 0u; k < inner; ++k) {
      right[col * inner + k] = static_cast<Value>(
          base * static_cast<Value>(1u + (col + 3u * k) % 5u));
    }
  }
  for (std::size_t col = 0u; col < cols; ++col) {
    for (std::size_t row = 0u; row < rows; ++row) {
      Wide sum = 0;
      for (std::size_t k = 0u; k < inner; ++k) {
        sum += (static_cast<Wide>(left[k * rows + row]) *
                static_cast<Wide>(right[col * inner + k])) >>
               fraction_bits;
      }
      expected[col * rows + row] = static_cast<Value>(sum);
    }
  }
  return RunMatrix(pick, rund::kernel::MatrixOp::Mul,
                   rund::kernel::MatrixShape{
                       .layout = rund::kernel::MatrixLayout::ColumnMajor,
                       .rows = rows,
                       .cols = cols,
                       .inner = inner,
                       .element_bytes = sizeof(Value),
                   },
                   left, right, expected);
}

} // namespace

[[nodiscard]] bool BackendRunsMatrix(const rund::AccelDevice &pick) {
  if (!NumericGeneratedSourcesCloseGenericDependencies()) {
    return false;
  }
  constexpr rund::kernel::i32 one = 0x40000000;
  constexpr rund::kernel::i32 zero = 0;
  constexpr rund::kernel::i32 precise_lhs = 0x40000001;
  constexpr rund::kernel::i32 precise_rhs = 0x40000003;
  constexpr rund::kernel::i32 precise_product = static_cast<rund::kernel::i32>(
      (static_cast<rund::kernel::i64>(precise_lhs) * precise_rhs) >> 31u);
  const bool mul = RunMatrix(
      pick, rund::kernel::MatrixOp::Mul,
      rund::kernel::MatrixShape{.rows = 2u,
                                .cols = 2u,
                                .inner = 2u,
                                .element_bytes = sizeof(rund::kernel::i32)},
      std::array<rund::kernel::i32, 4u>{one, zero, zero, one},
      std::array<rund::kernel::i32, 4u>{one, zero, zero, one},
      std::array<rund::kernel::i32, 4u>{0x20000000, zero, zero, 0x20000000});
  const bool transpose = RunMatrix(
      pick, rund::kernel::MatrixOp::Transpose,
      rund::kernel::MatrixShape{.rows = 2u,
                                .cols = 3u,
                                .inner = 0u,
                                .element_bytes = sizeof(rund::kernel::i32)},
      std::array<rund::kernel::i32, 6u>{1, 2, 3, 4, 5, 6},
      std::array<rund::kernel::i32, 1u>{},
      std::array<rund::kernel::i32, 6u>{1, 4, 2, 5, 3, 6});
  const bool batch = RunMatrix(
      pick, rund::kernel::MatrixOp::BatchMul,
      rund::kernel::MatrixShape{.rows = 1u,
                                .cols = 1u,
                                .inner = 1u,
                                .batch_count = 2u,
                                .element_bytes = sizeof(rund::kernel::i32)},
      std::array<rund::kernel::i32, 2u>{one, one},
      std::array<rund::kernel::i32, 2u>{one, one},
      std::array<rund::kernel::i32, 2u>{0x20000000, 0x20000000});
  const bool precise = RunMatrix(
      pick, rund::kernel::MatrixOp::Mul,
      rund::kernel::MatrixShape{.rows = 1u,
                                .cols = 1u,
                                .inner = 1u,
                                .element_bytes = sizeof(rund::kernel::i32)},
      std::array<rund::kernel::i32, 1u>{precise_lhs},
      std::array<rund::kernel::i32, 1u>{precise_rhs},
      std::array<rund::kernel::i32, 1u>{precise_product});
  constexpr rund::kernel::i64 wide_one = 0x4000000000000000ll;
  constexpr rund::kernel::i64 wide_zero = 0;
  const bool wide = RunMatrix(
      pick, rund::kernel::MatrixOp::Mul,
      rund::kernel::MatrixShape{.rows = 2u,
                                .cols = 2u,
                                .inner = 2u,
                                .element_bytes = sizeof(rund::kernel::i64)},
      std::array<rund::kernel::i64, 4u>{wide_one, wide_zero, wide_zero,
                                        wide_one},
      std::array<rund::kernel::i64, 4u>{wide_one, wide_zero, wide_zero,
                                        wide_one},
      std::array<rund::kernel::i64, 4u>{0x2000000000000000ll, wide_zero,
                                        wide_zero, 0x2000000000000000ll});
  const bool wide_bits = RunMatrix(
      pick, rund::kernel::MatrixOp::Transpose,
      rund::kernel::MatrixShape{.rows = 2u,
                                .cols = 2u,
                                .inner = 0u,
                                .element_bytes = sizeof(rund::kernel::i64)},
      std::array<rund::kernel::i64, 4u>{
          0x4000000000000001ll, 0x2000000000000003ll,
          static_cast<rund::kernel::i64>(0xc000000000000005ull), 7ll},
      std::array<rund::kernel::i64, 1u>{},
      std::array<rund::kernel::i64, 4u>{
          0x4000000000000001ll,
          static_cast<rund::kernel::i64>(0xc000000000000005ull),
          0x2000000000000003ll, 7ll});
  const bool tiled = RunTiledMatrix<rund::kernel::i32>(pick) &&
                     RunTiledMatrix<rund::kernel::i64>(pick);
  return mul && transpose && batch && precise && wide && wide_bits && tiled;
}

[[nodiscard]] bool MatrixRejectsShapeMismatch(const rund::AccelDevice &pick) {
  namespace fix = node_accel_contract::primitive;
  auto context = rund::node::accel::OpenAccel(pick);
  if (!context.check.ok) {
    return false;
  }
  auto left = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::ReadOnly,
                               sizeof(rund::kernel::i32), 1u));
  auto right = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::ReadOnly,
                               sizeof(rund::kernel::i32), 1u));
  auto output = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::WriteOnly,
                               sizeof(rund::kernel::i32), 1u));
  const std::array<rund::AccelGraphBufferRef, 3u> refs{
      rund::AccelGraphBufferRef{.buffer = &left,
                                .role = rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{.buffer = &right,
                                .role = rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{.buffer = &output,
                                .role = rund::kernel::BufferRole::Write},
  };
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelMatrix(
          refs.data(), refs.size(),
          rund::kernel::MatrixDesc{
              .op = rund::kernel::MatrixOp::Mul,
              .layout = rund::kernel::MatrixLayout::RowMajor,
              .arithmetic = rund::kernel::MatrixArithmetic::Fixed,
              .rows = 2u,
              .cols = 2u,
              .inner = 2u,
              .element_bytes = sizeof(rund::kernel::i32),
              .fixed_format = test::FixedFormatForLane(
                  rund::kernel::ComputeScalar::Lane32,
                  rund::kernel::ComputeApproximation::Deterministic),
          }),
  };
  const auto kernel = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = nodes.data(),
                   .node_count = nodes.size(),
                   .scalar = rund::kernel::ComputeScalar::Lane32,
                   .domain = rund::kernel::ComputeDomain::Fixed,
                   .fixed_format = test::FixedFormatForLane(
                       rund::kernel::ComputeScalar::Lane32),
               });
  return !kernel.check.ok;
}

[[nodiscard]] bool AvailableBackendsRunMatrixNatively() {
  namespace fix = node_accel_contract::primitive;
  for (const rund::AccelApi api :
       {rund::AccelApi::Metal, rund::AccelApi::Vulkan}) {
    const rund::AccelDevice pick =
        rund::node::accel::PickAccel(fix::Policy(api));
    if (!pick.check.ok) {
      continue;
    }
    if (pick.api == rund::AccelApi::Cpu) {
      return false;
    }
    if (!BackendRunsMatrix(pick)) {
      return false;
    }
  }
  return true;
}

} // namespace node_accel_contract
