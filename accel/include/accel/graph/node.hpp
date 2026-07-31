#pragma once

#include <accel/graph/buffer/ref.hpp>
#include <kernel/program/compute/compact/model.hpp>
#include <kernel/program/compute/factor/model.hpp>
#include <kernel/program/compute/gather/model.hpp>
#include <kernel/program/compute/graph/schema.hpp>
#include <kernel/program/compute/histogram/model.hpp>
#include <kernel/program/compute/ir.hpp>
#include <kernel/program/compute/matrix/model.hpp>
#include <kernel/program/compute/partition/model.hpp>
#include <kernel/program/compute/reduce/model.hpp>
#include <kernel/program/compute/scan/model.hpp>
#include <kernel/program/compute/scatter/model.hpp>
#include <kernel/program/compute/scatter/reduce/model.hpp>
#include <kernel/program/compute/segmented/reduce/model.hpp>
#include <kernel/program/compute/segmented/scan/model.hpp>
#include <kernel/program/compute/solve/model.hpp>
#include <kernel/program/compute/sort/model.hpp>
#include <kernel/program/compute/spectrum/model.hpp>
#include <kernel/program/compute/stencil/model.hpp>
#include <kernel/program/compute/transform/model.hpp>

#include <cstdint>

namespace rund {

struct AccelGraphNode {
  const rund::kernel::ComputeIR *ir = nullptr;
  const AccelGraphBufferRef *buffers = nullptr;
  std::uint64_t buffer_count = 0u;
  rund::kernel::NodeKind kind = rund::kernel::NodeKind::Map;
  std::uint64_t primitive_hash_hi = 0u;
  std::uint64_t primitive_hash_lo = 0u;
  std::uint64_t element_count = 0u;
  rund::kernel::GraphControl control{};
  rund::kernel::GraphSignature signature{};
  rund::kernel::SortDesc sort{};
  rund::kernel::CompactDesc compact{};
  rund::kernel::ScanDesc scan{};
  rund::kernel::SegmentedScanDesc segmented_scan{};
  rund::kernel::SegmentedReduceDesc segmented_reduce{};
  rund::kernel::GatherDesc gather{};
  rund::kernel::HistogramDesc histogram{};
  rund::kernel::PartitionDesc partition{};
  rund::kernel::ReduceDesc reduce{};
  rund::kernel::ScatterDesc scatter{};
  rund::kernel::ScatterReduceDesc scatter_reduce{};
  rund::kernel::StencilDesc stencil{};
  rund::kernel::TransformDesc transform{};
  rund::kernel::MatrixDesc matrix{};
  rund::kernel::FactorDesc factor{};
  rund::kernel::SolveDesc solve{};
  rund::kernel::SpectrumDesc spectrum{};
  bool barrier_before = false;
};

} // namespace rund
