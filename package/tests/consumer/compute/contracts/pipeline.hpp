#pragma once

#include "common.hpp"

template <class T>
concept HasProgramObservers =
    requires(const T &program, std::span<rund::compute::MemoryEntry> entries) {
      { program.valid() } -> std::same_as<bool>;
      {
        program.backend()
      } -> std::same_as<rund::compute::Result<rund::compute::Backend>>;
      { program.graph() } -> std::same_as<const rund::compute::graph::Info &>;
      {
        program.fingerprint()
      } -> std::same_as<rund::compute::graph::Fingerprint>;
      { program.memory() } -> std::same_as<rund::compute::MemoryStats>;
      {
        program.memory_snapshot(entries)
      } -> std::same_as<rund::compute::MemorySnapshot>;
    };

template <class T>
concept ConfiguresPipelineProfile = requires(T &builder) {
  {
    builder.profile(rund::compute::PipelineProfile::Steps)
  } -> std::same_as<T &>;
  {
    std::move(builder).profile(rund::compute::PipelineProfile::Steps)
  } -> std::same_as<T &&>;
};

template <class T>
concept ConfiguresPipelineSealedRepetitions = requires(T &builder) {
  { builder.template sealed_repetitions<1u>() } -> std::same_as<T &>;
  {
    std::move(builder).template sealed_repetitions<1024u>()
  } -> std::same_as<T &&>;
};

template <class T>
concept PlansPipeline = requires(T &builder) {
  {
    builder.plan()
  } -> std::same_as<rund::compute::Result<rund::compute::PipelinePlan>>;
  {
    builder.budget(rund::compute::MemoryBudget{.bytes = 1u})
  } -> std::same_as<T &>;
};

template <class T>
concept HasPipelinePlan = requires(const T &plan) {
  plan.persistent_bytes;
  plan.state_bytes;
  plan.state_bytes;
  plan.transient_bytes;
  plan.prepared_bytes;
  plan.residency.logical_bytes;
  plan.residency.page_bytes;
  plan.residency.page_count;
  plan.residency.frame_capacity;
  plan.residency.resident_bytes;
  plan.residency.epoch_count;
  plan.residency.identity_hi;
  plan.residency.identity_lo;
  plan.publish_bytes;
  plan.peak_bytes;
  plan.total_bytes;
  plan.allocation_count;
  plan.reuse_count;
  plan.publish_count;
  plan.largest_bytes;
  plan.largest_step;
  plan.largest_iteration;
  plan.largest_chunk;
  plan.view_bytes;
  plan.view_span_bytes;
  plan.view_backing_bytes;
  plan.view_offset_bytes;
  plan.view_stride_bytes;
  plan.view_element_bytes;
  plan.view_count;
  plan.view_alignment;
  plan.view_step;
  plan.view_iteration;
  plan.view_binding;
  plan.peak_step;
  plan.peak_iteration;
};

template <class T>
concept HasPipelineProfile = requires(
    const T &pipeline, std::span<rund::compute::PipelineStepProfile> steps) {
  {
    pipeline.profile(steps)
  } -> std::same_as<
      rund::compute::Result<rund::compute::PipelineProfileSnapshot>>;
};

template <class T>
concept HasPipelineStepStats = requires(const T &stats) {
  stats.sample_count;
  stats.original_dispatches;
  stats.final_dispatches;
  stats.barrier_count;
  stats.tile_count;
  stats.tile_size;
  stats.vector_chunks;
  stats.tail_chunks;
  stats.workgroup_count;
  stats.work_item_count;
  stats.control;
  stats.worker_count;
  stats.participating_workers;
  { stats.available() } -> std::same_as<bool>;
};

template <class T>
concept HasStepTiming = requires(const T &timing) {
  timing.duration_ns;
  timing.sample_count;
  timing.clock;
  timing.relation;
  { timing.available() } -> std::same_as<bool>;
  { timing.saturated() } -> std::same_as<bool>;
};

template <class T>
concept HasPipelineStepProfile = requires(const T &step) {
  step.index;
  step.iteration;
  step.iteration_bound;
  step.program;
  step.timing;
  step.execution;
  step.memory;
};

template <class T>
concept HasPipelineProfileSnapshot = requires(const T &snapshot) {
  snapshot.execution;
  snapshot.memory;
  snapshot.shared_memory;
  snapshot.observation;
  snapshot.referenced_resource_bytes;
  snapshot.instrumentation_command_count;
  snapshot.instrumentation_byte_count;
  snapshot.written;
  snapshot.total;
  { snapshot.truncated() } -> std::same_as<bool>;
};
