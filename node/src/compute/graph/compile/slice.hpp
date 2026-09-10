#pragma once

#include "../../program/state.hpp"


#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rund::compute::detail::graph_compile {

enum class SliceResourceKind : std::uint8_t {
  ExternalInput,
  Internal,
  ExternalOutput,
};

struct TiledGraphSliceResource final {
  std::uint32_t resource{};
  Type type{Type::I32};
  FixedFormat format{};
  std::size_t count{};
  SliceResourceKind kind{SliceResourceKind::Internal};
};

struct TiledGraphStageSlice final {
  std::shared_ptr<ProgramState> program;
  std::vector<std::uint32_t> inputs;
  std::vector<std::uint32_t> outputs;
  std::uint32_t node{};
  bool tile_partial{};
};

// Each canonical stage is independently recompiled. Original Graph value IDs
// remain the resource IDs, so planner liveness and the physical intermediate
// actually consumed by the prepared stage cannot diverge.
struct TiledGraphSlices final {
  std::vector<TiledGraphSliceResource> resources;
  std::vector<TiledGraphStageSlice> stages;
  // Optional service-free semantic owner for a public-input Map DAG whose
  // physical stages must remain distinct for Host wavefront residency.
  std::shared_ptr<ProgramState> service_free_prefix;
  std::size_t capacity{};
  // Canonical Program-port order. Each row remains an independently backed
  // Graph resource even when the service-free semantic Map consumes them in
  // one native dispatch.
  std::vector<std::uint32_t> input_resources;
  std::uint32_t output_resource{};
};

// Lowers a pure U64 pointwise Map DAG followed by one U64 reduction.
// Linear/internal-only prefixes remain one fused physical stage. A prefix
// whose later Map rereads the public external input keeps those Maps as exact
// independently compiled stages so the sealed ready wavefront, next-use pins,
// and physical reuse authority remain observable, while an optional strongly
// owned semantic Program preserves the exact service-free expression DAG.
// Unsupported topology fails closed; no stage is projected from an
// already-compiled opaque Program.
[[nodiscard]] Result<TiledGraphSlices>
compile_tiled_graph_slices(const std::shared_ptr<ProgramState> &program);

// Preserves every authored Map stage for the bounded Graph wavefront when the
// service-free pointwise compiler cannot prove a whole-run fused recurrence.
// The terminal resource is a true page-sized public output, not a reduction
// partial.
[[nodiscard]] Result<TiledGraphSlices> compile_tiled_graph_pointwise_slices(
    const std::shared_ptr<ProgramState> &program);

} // namespace rund::compute::detail::graph_compile
