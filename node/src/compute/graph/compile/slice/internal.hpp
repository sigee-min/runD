#pragma once

#include "../slice.hpp"

#include "../../state.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rund::compute::detail::graph_compile::slice_detail {

using SliceResult = Result<TiledGraphSlices>;

struct SliceSource final {
  const GraphState *graph{};
  const GraphPrimitive *reduce{};
  Type type{Type::U64};
  std::size_t capacity{};
};

[[nodiscard]] Result<SliceSource>
inspect_reduce(const std::shared_ptr<ProgramState> &) noexcept;
[[nodiscard]] Result<SliceSource>
inspect_pointwise(const std::shared_ptr<ProgramState> &) noexcept;

[[nodiscard]] bool bind_count(GraphState &, std::uint32_t,
                              std::size_t capacity);
[[nodiscard]] bool requires_external_wavefront(SliceSource) noexcept;
[[nodiscard]] Result<std::shared_ptr<ProgramState>>
compile_map_prefix(SliceSource, std::vector<std::uint32_t> &inputs,
                   std::vector<std::uint32_t> &outputs);
[[nodiscard]] Result<std::shared_ptr<ProgramState>>
compile_map_stage(SliceSource, std::size_t stage,
                  std::vector<std::uint32_t> &inputs,
                  std::vector<std::uint32_t> &outputs);
[[nodiscard]] Result<std::shared_ptr<ProgramState>> compile_reduce(SliceSource);

void append_referenced_resources(TiledGraphSlices &, const GraphState &);

} // namespace rund::compute::detail::graph_compile::slice_detail
