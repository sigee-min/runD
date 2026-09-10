#pragma once

#include "../../recurrence.hpp"
#include "../match.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace rund::node::accel::detail::nested_aggregate_detail {

using rund::kernel::BindingSet;
using rund::kernel::ResidentBindingRange;
using rund::kernel::ResidentBufferRef;

struct View final {
  const ResidentBufferRef *ref{};
  const std::shared_ptr<void> *handle{};

  [[nodiscard]] bool valid() const noexcept;
};

struct SeedProjection final {
  BackendRead queue{};
  BackendRead domain{};
  BackendRead count{};
  View tile_low{};
  View tile_status{};
  View tile_state{};
  View tile_count{};
  std::uint32_t invalid_index_source_node{NoNode};
  std::uint32_t reduce_overflow_source_node{NoNode};
};

// Cold-only identity used while proving that every authored template belongs
// to one admitted program. It is deliberately not retained in the aggregate:
// native execution consumes the normalized proof, never an execution owner.
struct ProgramFingerprint final {
  rund::AccelApi api{rund::AccelApi::Auto};
  std::uint64_t graph_id_hi{};
  std::uint64_t graph_id_lo{};
  rund::kernel::ComputeScalar scalar{rund::kernel::ComputeScalar::Lane32};
  rund::kernel::ComputeDomain domain{rund::kernel::ComputeDomain::Fixed};
  rund::kernel::ComputeFixedFormat fixed_format{};
};

[[nodiscard]] View At(const ResidentBindingRange &range,
                      std::uint64_t index) noexcept;
[[nodiscard]] View At(const ResidentBufferRef *ref,
                      const std::shared_ptr<void> *handle) noexcept;
[[nodiscard]] bool SameStorage(View left, View right) noexcept;
[[nodiscard]] bool ReadView(View view) noexcept;
[[nodiscard]] bool WriteView(View view) noexcept;
[[nodiscard]] bool U32View(View view, std::uint64_t count) noexcept;
[[nodiscard]] bool DenseU32Workspace(View view, std::uint64_t count) noexcept;
[[nodiscard]] bool InternalOutput(const BackendRun &run,
                                  const BoundStep &step) noexcept;
[[nodiscard]] bool DisjointStorage(View left, View right) noexcept;
[[nodiscard]] NestedAggregateWorkspace Workspace(View view);
[[nodiscard]] BackendRead Read(View view);
[[nodiscard]] bool SameRead(const BackendRead &left,
                            const BackendRead &right) noexcept;
[[nodiscard]] bool ReadyRun(const BackendRun *run,
                            std::size_t step_count) noexcept;
[[nodiscard]] bool ReadyStep(const BoundStep &step,
                             rund::kernel::NodeKind kind, bool first) noexcept;
[[nodiscard]] ProgramFingerprint ProgramIdentity(const BackendRun &run) noexcept;
[[nodiscard]] bool SameProgram(const BackendRun &run,
                               const ProgramFingerprint &identity) noexcept;
[[nodiscard]] bool U32Program(const ProgramFingerprint &identity) noexcept;

[[nodiscard]] bool MapShape(const BoundStep &step, MapSemanticKind semantic,
                            std::uint64_t inputs, std::uint64_t outputs,
                            BindingSet &bindings) noexcept;
[[nodiscard]] bool CollectiveStep(const BoundStep &step,
                                  rund::kernel::NodeKind kind) noexcept;
[[nodiscard]] bool GatherShape(const BoundStep &step, std::uint32_t tile,
                               View &values, View &indices, View &count,
                               View &output) noexcept;
[[nodiscard]] bool ReduceShape(const BoundStep &step, std::uint32_t tile,
                               View &input, View &count,
                               View &output) noexcept;
[[nodiscard]] bool SeedShape(const BackendRun &run, const BackendWindow &window,
                             SeedProjection &projection,
                             const char *&reason);

[[nodiscard]] bool ExactAggregateEnvelope(
    std::span<const BackendBatchEntry> templates,
    std::span<const std::uint8_t> barriers, std::size_t first,
    NestedTemplateGeometry &geometry) noexcept;
[[nodiscard]] bool ScalarMap(const BackendRun &run,
                             const ProgramFingerprint &program,
                             MapSemanticKind semantic, std::uint64_t inputs,
                             BindingSet &bindings,
                             std::uint64_t final_dispatches) noexcept;
[[nodiscard]] bool BuildAction(
    std::span<const BackendBatchEntry> templates,
    const NestedTemplateShape &shape, const SeedProjection &seed,
    ProgramFingerprint &program, NestedScalarExpr &expression, View &final_state,
    std::uint32_t &dispatches) noexcept;
[[nodiscard]] bool BuildFold(
    std::span<const BackendBatchEntry> templates,
    const NestedTemplateShape &shape, const SeedProjection &seed,
    View action_state, const BackendPublish &publication,
    ProgramFingerprint &program, NestedScalarExpr &expression,
    std::uint32_t &dispatches) noexcept;
[[nodiscard]] bool PublicationFor(
    std::span<const BackendPublish> publications, const BackendWindow &window,
    BackendPublish &publication, std::uint32_t &index) noexcept;

} // namespace rund::node::accel::detail::nested_aggregate_detail
