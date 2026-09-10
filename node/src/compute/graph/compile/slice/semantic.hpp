#pragma once

#include "../../../program/state.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace rund::compute::detail {

struct GraphState;

namespace graph_compile {

// Seven public reads plus the single public output exactly fill the fixed
// DeviceVsm resident table.  Keep this semantic boundary tied to the public
// Virtual input authority rather than a smaller test-shaped subset.
inline constexpr std::size_t ServiceFreeMapScanInputCapacity = 7u;

// Composes a pure Map DAG whose public leaves are the exact ordered Graph
// interface inputs. All leaves and the terminal currently share one U32/U64
// type; the returned controlled Program retains every input ordinal.
[[nodiscard]] Result<std::shared_ptr<ProgramState>>
compile_service_free_map_semantic_inputs(const GraphState &, Type, std::size_t,
                                         std::span<const std::uint32_t> inputs,
                                         std::uint32_t terminal_value);

// Rebuilds a canonical sole-input/sole-output pure U32/U64 Map DAG as one
// unbounded Map step. The returned Program owns the composed expression and
// therefore has no authored intermediate residency stages.
[[nodiscard]] Result<std::shared_ptr<ProgramState>>
compile_service_free_map_program(const std::shared_ptr<ProgramState> &);

// Rebuilds a canonical one-through-seven-input pure-Map DAG followed by one
// unbounded U64 Scan as one Map step plus one Scan step. The composed Map
// expression is owned by the returned Program; authored intermediate Graph
// values do not become physical residency stages.
[[nodiscard]] Result<std::shared_ptr<ProgramState>>
compile_service_free_map_scan(const std::shared_ptr<ProgramState> &);

} // namespace graph_compile
} // namespace rund::compute::detail
