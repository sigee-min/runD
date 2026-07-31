#pragma once

#include "../model.hpp"

#include <accel/graph/node.hpp>
#include <kernel/program/compute/factor/model.hpp>
#include <kernel/program/compute/matrix/model.hpp>
#include <kernel/program/compute/reduce/model.hpp>
#include <kernel/program/compute/scatter/reduce/model.hpp>
#include <kernel/program/compute/segmented/scan/model.hpp>
#include <kernel/program/compute/spectrum/model.hpp>
#include <kernel/program/compute/stencil/model.hpp>
#include <kernel/program/compute/transform/model.hpp>

#include <cstdint>
#include <optional>
#include <span>

namespace rund::compute::detail::graph_build_detail {

[[nodiscard]] std::optional<kernel::ReduceOp>
reduce_op(std::uint32_t mode, bool count = false) noexcept;

[[nodiscard]] std::optional<kernel::ScatterReduceOp>
scatter_reduce_op(std::uint32_t mode) noexcept;

[[nodiscard]] std::optional<kernel::StencilOp>
stencil_op(std::uint32_t mode) noexcept;

[[nodiscard]] std::optional<kernel::FactorOp>
factor_op(std::uint32_t mode) noexcept;

[[nodiscard]] std::optional<kernel::SegmentedScanOp>
segmented_scan_op(std::uint32_t mode) noexcept;

[[nodiscard]] std::optional<kernel::TransformDir>
transform_direction(std::uint32_t mode) noexcept;

[[nodiscard]] std::optional<kernel::MatrixOp>
matrix_op(std::uint32_t mode) noexcept;

[[nodiscard]] std::optional<kernel::SpectrumOp>
spectrum_op(std::uint32_t mode) noexcept;

[[nodiscard]] std::optional<kernel::SpectrumVectors>
spectrum_vectors(std::uint64_t mode) noexcept;

[[nodiscard]] const char *unsupported(Primitive primitive,
                                      PrimitiveOptions options) noexcept;

[[nodiscard]] rund::AccelGraphNode make_node(Primitive primitive,
                                             std::span<const GraphArg> inputs,
                                             PrimitiveOptions options,
                                             std::uint32_t &primary_write);

} // namespace rund::compute::detail::graph_build_detail
