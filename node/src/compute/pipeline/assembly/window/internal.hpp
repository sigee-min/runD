#pragma once

#include "../../state.hpp"
#include "../internal.hpp"

#include <rund/compute/abi/state.hpp>
#include <rund/compute/abi/resource.hpp>
#include <rund/compute/status.hpp>
#include <rund/compute/pipeline/shape.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rund::compute::detail {

// The nested append is one transaction.  This value object records the
// pre-append sizes and owns the only rollback path for all build containers.
// Its implementation is compiled in mutation.cpp so phase headers carry no
// algorithm body or second mutation policy.
class PipelineBuildMutation final {
public:
  explicit PipelineBuildMutation(PipelineBuildState &) noexcept;
  PipelineBuildMutation(const PipelineBuildMutation &) = delete;
  PipelineBuildMutation &operator=(const PipelineBuildMutation &) = delete;
  ~PipelineBuildMutation() noexcept;

  void fail(Reason) noexcept;
  void commit() noexcept;

private:
  PipelineBuildState &build_;
  std::size_t steps_{};
  std::size_t bindings_{};
  std::size_t internals_{};
  std::size_t publications_{};
  std::size_t window_controls_{};
  std::size_t nested_windows_{};
  bool committed_{};
};

// Borrowed request view used only while append_pipeline_window_repeat is
// running.  It contains no retained or projected build state.
struct WindowAssemblyInput final {
  const std::shared_ptr<ProgramState> *seed{};
  const std::shared_ptr<ProgramState> *action{};
  const std::shared_ptr<ProgramState> *fold{};
  const ResourceView *resident{};
  std::span<const ResourceView> inputs{};
  std::span<const ResourceView> final_outputs{};
  std::span<const ResourceView> window_outputs{};
  std::size_t maximum{};
  std::size_t tile{};
  std::size_t inner{};
  std::size_t terminal{};
  std::uint32_t expected{};
};

// Validation produces immutable facts consumed by all later phases.  The
// source Programs and authored Views remain owned by their callers/build.
struct WindowAssemblyCounts final {
  node::accel::detail::NestedTemplateShape nested_shape{};
  std::size_t seed_output_count{};
  std::size_t action_output_count{};
  std::size_t fold_output_count{};
  std::size_t recurrent_count{};
  std::size_t window_count{};
  std::size_t seed_external_count{};
  std::size_t binding_count{};
};

// Bounded scratch bindings are projections into build->internals or authored
// resources.  They are discarded after the transaction and never become a
// parallel authority.
struct WindowAssemblyResources final {
  std::vector<PipelineBinding> outer_seed;
  std::vector<PipelineBinding> seed_external;
  std::vector<PipelineBinding> outer_first;
  std::vector<PipelineBinding> outer_second;
  std::vector<PipelineBinding> final;
  std::vector<PipelineBinding> window_tile;
  std::vector<PipelineBinding> window_target;
  std::vector<PipelineBinding> tile_first;
  std::vector<PipelineBinding> tile_second;
  std::uint32_t ordinal_owner{};
  std::uint16_t nested{};
  PipelineBuildWindowControlOrdinal window_control{};
};

[[nodiscard]] bool validate_window_request(PipelineBuildState &,
                                           const WindowAssemblyInput &,
                                           WindowAssemblyCounts &);

[[nodiscard]] bool materialize_window_resources(PipelineBuildState &,
                                                const WindowAssemblyInput &,
                                                const WindowAssemblyCounts &,
                                                WindowAssemblyResources &,
                                                PipelineBuildMutation &);

[[nodiscard]] bool emit_window_steps(PipelineBuildState &,
                                     const WindowAssemblyInput &,
                                     const WindowAssemblyCounts &,
                                     const WindowAssemblyResources &,
                                     PipelineBuildMutation &);

[[nodiscard]] bool publish_window(PipelineBuildState &,
                                  const WindowAssemblyInput &,
                                  const WindowAssemblyCounts &,
                                  const WindowAssemblyResources &);

} // namespace rund::compute::detail
