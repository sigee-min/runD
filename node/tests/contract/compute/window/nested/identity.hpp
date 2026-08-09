#pragma once

#include "src/compute/job/state.hpp"
#include "src/compute/pipeline/state.hpp"
#include <rund/compute.hpp>
#include <rund/compute/pipeline.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rund::node::test_contract::window {

struct JobViewIdentity final {
  std::size_t offset{};
  std::size_t count{};
  std::size_t stride{};
  std::size_t element_bytes{};
  std::size_t alignment{};
  [[nodiscard]] bool
  operator==(const JobViewIdentity &) const noexcept = default;
};

struct KernelViewIdentity final {
  std::uint64_t binding{};
  std::size_t slot{};
  std::uint64_t backing_bytes{};
  std::uint64_t offset_bytes{};
  std::uint64_t count{};
  std::uint64_t stride_bytes{};
  std::uint64_t element_bytes{};
  std::uint32_t usage{};
  [[nodiscard]] bool
  operator==(const KernelViewIdentity &) const noexcept = default;
};

struct ArenaSlotIdentity final {
  std::size_t words{};
  std::size_t owner{};
  std::size_t offset_words{};
  [[nodiscard]] bool
  operator==(const ArenaSlotIdentity &) const noexcept = default;
};

struct ScratchIdentity final {
  std::size_t slot{};
  std::uint64_t bytes{};
  [[nodiscard]] bool
  operator==(const ScratchIdentity &) const noexcept = default;
};

struct ResidentBindingIdentity final {
  const void *handle{};
  std::uint64_t id{};
  std::uint64_t bytes{};
  std::uint64_t offset_bytes{};
  std::uint64_t element_bytes{};
  std::uint64_t stride_bytes{};
  std::uint64_t count{};
  std::uint32_t usage{};
  [[nodiscard]] bool
  operator==(const ResidentBindingIdentity &) const noexcept = default;
};

struct CpuTransferIdentity final {
  const rund::compute::detail::BufferState *external{};
  const rund::compute::detail::BufferState *staging{};
  JobViewIdentity view{};
  std::uint32_t binding{};
  [[nodiscard]] bool
  operator==(const CpuTransferIdentity &) const noexcept = default;
};

struct JobBindingIdentity final {
  const rund::compute::detail::JobState *owner{};
  const rund::compute::detail::ProgramState *program{};
  rund::compute::graph::Fingerprint program_fingerprint{};
  const rund::compute::detail::JobWorkspace *workspace{};
  const rund::compute::detail::ProgramState *workspace_program{};
  const rund::compute::detail::JobArena *arena{};
  const void *prepared_owner{};
  const void *write_prepared_owner{};
  bool prepared_ok{};
  bool write_prepared_ok{};
  bool arena_bound{};
  bool arena_binds_heap{};
  bool arena_binds_ok{};
  std::vector<const rund::compute::detail::BufferState *> inputs;
  std::vector<const rund::compute::detail::BufferState *> write_inputs;
  std::vector<const rund::compute::detail::BufferState *> graph_buffers;
  std::vector<const rund::compute::detail::BufferState *> effective_graph;
  std::vector<const rund::compute::detail::BufferState *> outputs;
  std::vector<const rund::compute::detail::BufferState *> workspace_buffers;
  std::vector<const rund::compute::detail::BufferState *> arena_buffers;
  std::vector<std::size_t> workspace_offsets;
  std::vector<JobViewIdentity> input_views;
  std::vector<JobViewIdentity> output_views;
  std::vector<KernelViewIdentity> kernel_views;
  std::vector<CpuTransferIdentity> cpu_inputs;
  std::vector<CpuTransferIdentity> cpu_outputs;
  std::vector<ArenaSlotIdentity> arena_slots;
  std::vector<ResidentBindingIdentity> arena_bindings;
  std::vector<ScratchIdentity> scratch;
  [[nodiscard]] bool
  operator==(const JobBindingIdentity &) const noexcept = default;
};

struct StepBindingIdentity final {
  const rund::compute::detail::ProgramState *program{};
  const rund::compute::detail::JobState *normal{};
  const rund::compute::detail::JobState *alternate{};
  rund::compute::detail::PipelineRoute route{
      rund::compute::detail::PipelineRoute::Ordinary};
  std::uint32_t logical_step{};
  std::uint32_t iteration{};
  std::uint16_t window{};
  [[nodiscard]] bool
  operator==(const StepBindingIdentity &) const noexcept = default;
};

struct PipelineBindingIdentity final {
  const rund::compute::detail::PipelineState *owner{};
  const void *prepared_owner{};
  const void *alternate_prepared_owner{};
  bool prepared_ok{};
  bool alternate_prepared_ok{};
  bool transactional{};
  bool valid{};
  std::vector<StepBindingIdentity> steps;
  std::vector<const rund::compute::detail::JobState *> normal_jobs;
  std::vector<const rund::compute::detail::JobState *> alternate_jobs;
  std::vector<JobBindingIdentity> jobs;
  std::vector<const rund::compute::detail::BufferState *> resources;
  std::vector<const rund::compute::detail::BufferState *> prepared_buffers;
  std::vector<const rund::compute::detail::BufferState *> claims;
  std::vector<const rund::compute::detail::BufferState *> alternate_claims;
  std::vector<const rund::compute::detail::BufferState *> state_banks;
  std::vector<const rund::compute::detail::BufferState *> publications;
  std::vector<const rund::compute::detail::BufferState *> window_counts;
  [[nodiscard]] bool
  operator==(const PipelineBindingIdentity &) const noexcept = default;
};

[[nodiscard]] PipelineBindingIdentity
CaptureBindingIdentity(const rund::compute::Pipeline &, rund::compute::Backend);

} // namespace rund::node::test_contract::window
