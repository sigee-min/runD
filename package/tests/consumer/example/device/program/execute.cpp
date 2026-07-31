#include "model.hpp"

#include "../../../support/allocation.hpp"

namespace package_device_program {

[[nodiscard]] int RunTick(Device &device, const Backend expected_backend,
                          const PipelineProfile profile_mode,
                          Evidence &evidence) {
  const auto reject = [expected_backend, profile_mode](const char *phase) {
    std::fprintf(stderr, "installed pipeline phase=%s backend=%u profile=%u\n",
                 phase, static_cast<unsigned>(expected_backend),
                 static_cast<unsigned>(profile_mode));
    return 2;
  };
  const auto selected = device.backend();
  if (!selected)
    return selected.exit_code();
  if (*selected != expected_backend)
    return reject("backend");
  if (const int contract = CheckFailures(device, *selected); contract != 0) {
    return contract;
  }
  const std::uint64_t authoring_calls_before_compile = domain_authoring_calls;

  // One installed Flow demonstrates the bounded-program basis: exact state,
  // active-set compaction, count-aware indexed resident load, bounded emit,
  // finite worklist iteration, and deterministic conflict resolution.
  auto tick =
      on(device)
          .input<std::int32_t>(Capacity)
          .branch([](auto state) {
            ++domain_authoring_calls;
            // 1D simulation normal form. Every callback below builds
            // symbolic Flow once; none is retained as a host callback.
            auto spatial_state = state.map("spatial-update", [](auto value) {
              return value - std::int32_t{1};
            });
            // `compact` returns ordered source indices for source-lineage
            // lookup; bounded `indices()` returns local worklist ordinals.
            auto broadphase_indices =
                spatial_state
                    .map("broadphase-mask",
                         [](auto value) {
                           return select(value > std::int32_t{0},
                                         std::uint32_t{1}, std::uint32_t{0});
                         })
                    .compact(Compact{.capacity = Capacity});
            auto broadphase_order =
                spatial_state.gather(broadphase_indices).sort();
            auto candidates = broadphase_order.expand(
                MaxItems{CandidateCapacity / Capacity},
                [](auto value) {
                  return select(value > std::int32_t{3}, std::int32_t{2},
                                std::int32_t{1});
                },
                [](auto value, auto ordinal) { return value + ordinal; });
            auto contacts = candidates.filter(
                [](auto value) { return value > std::int32_t{1}; });
            auto island_keys =
                contacts.indices().map("island-key", [](auto ordinal) {
                  return ordinal & std::uint32_t{1};
                });
            auto island_work =
                contacts.scatter_reduce(island_keys, IslandCount, Reduce::Sum);
            auto solver_seed = island_work.filter(
                [](auto value) { return value > std::int32_t{1}; });
            auto compact_events = solver_seed.template unroll<2u>(
                [](auto work) {
                  return work.map("solver-step", [](auto value) {
                    return value - std::int32_t{1};
                  });
                },
                [](auto value) { return value <= std::int32_t{1}; });
            auto solver_impulse = compact_events.reduce(Reduce::Sum);
            auto integrated_state = spatial_state.combine(
                "integrate-state", solver_impulse,
                [](auto value, auto impulse) {
                  return value + (impulse & std::int32_t{1});
                });
            auto state_hash =
                integrated_state
                    .map("hash-state",
                         [](auto value) { return rund::compute::hash(value); })
                    .reduce(Reduce::Sum);
            return outputs(integrated_state, compact_events, island_work,
                           state_hash);
          })
          .compile();
  // Bounded<T> is a Program-boundary authoring schema. Its resident ABI is
  // ordinary (values, count), so Pipeline carries the device count directly
  // into controlled work without a host observation or wrapper allocation.
  auto integrate_events =
      on(device)
          .input<Bounded<std::int32_t>>(EventCapacity)
          .branch([](auto events) {
            ++domain_authoring_calls;
            return events.map("persist-events", [](auto value) {
              return value * std::int32_t{2};
            });
          })
          .compile();
  if (!tick)
    return tick.exit_code();
  if (!integrate_events)
    return integrate_events.exit_code();
  const std::uint64_t compiled_authoring_calls = domain_authoring_calls;
  if (compiled_authoring_calls != authoring_calls_before_compile + 2u) {
    return reject("author");
  }

  constexpr std::array<std::int32_t, Capacity> initial{8, -2, 7,  0,
                                                       6, 1,  -1, 5};
  auto state = device.upload<std::int32_t>(initial);
  auto pending_state = device.buffer<std::int32_t>(Capacity);
  auto events = device.buffer<std::int32_t>(EventCapacity);
  auto event_count = device.buffer<std::uint32_t>(1u);
  auto resolved = device.buffer<std::int32_t>(2u);
  auto state_hash = device.buffer<std::int32_t>(1u);
  auto integrated = device.buffer<std::int32_t>(EventCapacity);
  auto integrated_count = device.buffer<std::uint32_t>(1u);
  if (!state)
    return state.exit_code();
  if (!pending_state)
    return pending_state.exit_code();
  if (!events)
    return events.exit_code();
  if (!event_count)
    return event_count.exit_code();
  if (!resolved)
    return resolved.exit_code();
  if (!state_hash)
    return state_hash.exit_code();
  if (!integrated)
    return integrated.exit_code();
  if (!integrated_count)
    return integrated_count.exit_code();
  auto prepared = pipeline(device)
                      .profile(profile_mode)
                      .state(*state, *pending_state)
                      .then(*tick, read(*state),
                            write(*pending_state, *events, *event_count,
                                  *resolved, *state_hash))
                      .then(*integrate_events, read(*events, *event_count),
                            write(*integrated, *integrated_count))
                      .commit()
                      .prepare();
  if (!prepared) {
    return prepared.exit_code();
  }
  Pipeline plan = std::move(prepared).value();
  if (plan.generation() != 0u)
    return reject("prepare:generation");
  std::array<MemoryEntry, MemoryEntryCapacity> prepared_entries{};
  std::array<MemoryEntry, MemoryEntryCapacity> first_entries{};
  std::array<MemoryEntry, MemoryEntryCapacity> second_entries{};
  const MemorySnapshot prepared_memory = plan.memory_snapshot(prepared_entries);
  if (!prepared_memory.summary.available() || prepared_memory.truncated() ||
      prepared_memory.summary.backend != expected_backend ||
      prepared_memory.summary.scope != MemoryScope::Pipeline) {
    return reject("prepare:memory");
  }
  const bool count_allocations = expected_backend == Backend::Cpu;
  if (count_allocations)
    package_consumer::allocation::start();
  const Status first_tick = plan.run();
  if (!first_tick) {
    if (count_allocations)
      (void)package_consumer::allocation::stop();
    return first_tick.exit_code();
  }
  const MemorySnapshot first_memory = plan.memory_snapshot(first_entries);
  if (plan.generation() != 1u ||
      domain_authoring_calls != compiled_authoring_calls) {
    if (count_allocations)
      (void)package_consumer::allocation::stop();
    return reject("run:first");
  }
  const Status second_tick = plan.run();
  if (!second_tick) {
    if (count_allocations)
      (void)package_consumer::allocation::stop();
    return second_tick.exit_code();
  }
  const MemorySnapshot second_memory = plan.memory_snapshot(second_entries);
  std::array<PipelineStepProfile, 2u> profile_rows{};
  PipelineProfileSnapshot profile_snapshot{};
  if (profile_mode == PipelineProfile::Steps) {
    const auto profiled = plan.profile(profile_rows);
    if (!profiled) {
      if (count_allocations)
        (void)package_consumer::allocation::stop();
      return profiled.exit_code();
    }
    profile_snapshot = *profiled;
  } else {
    const auto unavailable_profile = plan.profile(profile_rows);
    if (unavailable_profile ||
        unavailable_profile.reason() != Reason::ProfileUnavailable) {
      if (count_allocations)
        (void)package_consumer::allocation::stop();
      return reject("profile:disabled");
    }
  }
  const std::uint64_t warm_allocations =
      count_allocations ? package_consumer::allocation::stop() : 0u;
  if (plan.generation() != 2u ||
      domain_authoring_calls != compiled_authoring_calls) {
    return reject("run:second");
  }
  if (warm_allocations != 0u ||
      !same_memory_snapshot(prepared_memory, prepared_entries, first_memory,
                            first_entries) ||
      !same_memory_snapshot(first_memory, first_entries, second_memory,
                            second_entries)) {
    std::fprintf(
        stderr,
        "installed pipeline memory allocations=%llu entries=%zu/%zu/%zu "
        "device=%llu/%llu/%llu resident=%llu/%llu/%llu\n",
        static_cast<unsigned long long>(warm_allocations),
        prepared_memory.written, first_memory.written, second_memory.written,
        static_cast<unsigned long long>(prepared_memory.summary.device.current),
        static_cast<unsigned long long>(first_memory.summary.device.current),
        static_cast<unsigned long long>(second_memory.summary.device.current),
        static_cast<unsigned long long>(
            prepared_memory.summary.resident.current),
        static_cast<unsigned long long>(first_memory.summary.resident.current),
        static_cast<unsigned long long>(
            second_memory.summary.resident.current));
    return reject("memory:stable");
  }
  if (profile_mode == PipelineProfile::Steps &&
      !valid_profile(
          expected_backend, profile_snapshot, profile_rows,
          std::array{tick->fingerprint(), integrate_events->fingerprint()},
          plan.memory(), 100u)) {
    std::fprintf(
        stderr,
        "installed complex profile backend=%u rows=%zu/%zu referenced=%llu "
        "commands=%llu bytes=%llu barriers=%llu row0=%llu/%llu row1=%llu/%llu "
        "timing=%u/%u\n",
        static_cast<unsigned>(expected_backend), profile_snapshot.written,
        profile_snapshot.total,
        static_cast<unsigned long long>(
            profile_snapshot.referenced_resource_bytes),
        static_cast<unsigned long long>(
            profile_snapshot.instrumentation_command_count),
        static_cast<unsigned long long>(
            profile_snapshot.instrumentation_byte_count),
        static_cast<unsigned long long>(
            profile_snapshot.execution.pipeline.barrier_count),
        static_cast<unsigned long long>(
            profile_rows[0].execution.original_dispatches),
        static_cast<unsigned long long>(
            profile_rows[0].execution.final_dispatches),
        static_cast<unsigned long long>(
            profile_rows[1].execution.original_dispatches),
        static_cast<unsigned long long>(
            profile_rows[1].execution.final_dispatches),
        static_cast<unsigned>(profile_rows[0].timing.clock),
        static_cast<unsigned>(profile_rows[1].timing.clock));
    return reject("profile");
  }
  // This snapshot is taken before any explicit output observation. The
  // second tick therefore proves the prepared warm path itself, rather than
  // accidentally charging an application-requested read to execution.
  const Stats warm = plan.stats();
  if (warm.backend != expected_backend || warm.pipeline_compiles != 0u ||
      warm.buffer_allocations != 0u || warm.descriptor_pool_creations != 0u ||
      warm.descriptor_set_allocations != 0u || warm.uploaded_bytes != 0u ||
      warm.download_events != 0u || warm.downloaded_bytes != 0u ||
      warm.publication.generation != 2u ||
      warm.publication.commit_count != 2u ||
      warm.control.generated_item_count == 0u ||
      warm.control.indirect_dispatch_count == 0u ||
      warm.control.indirect_work_item_count == 0u ||
      warm.control.iteration_count != 2u || warm.control.conflict_count == 0u) {
    std::fprintf(
        stderr,
        "installed pipeline telemetry backend=%u compile=%llu allocation=%llu "
        "pool=%llu descriptor=%llu upload=%llu download=%llu/%llu "
        "publication=%llu/%llu control=%llu/%llu/%llu/%llu/%llu/%llu\n",
        static_cast<unsigned>(warm.backend),
        static_cast<unsigned long long>(warm.pipeline_compiles),
        static_cast<unsigned long long>(warm.buffer_allocations),
        static_cast<unsigned long long>(warm.descriptor_pool_creations),
        static_cast<unsigned long long>(warm.descriptor_set_allocations),
        static_cast<unsigned long long>(warm.uploaded_bytes),
        static_cast<unsigned long long>(warm.download_events),
        static_cast<unsigned long long>(warm.downloaded_bytes),
        static_cast<unsigned long long>(warm.publication.generation),
        static_cast<unsigned long long>(warm.publication.commit_count),
        static_cast<unsigned long long>(warm.control.generated_item_count),
        static_cast<unsigned long long>(warm.control.indirect_dispatch_count),
        static_cast<unsigned long long>(warm.control.indirect_work_item_count),
        static_cast<unsigned long long>(warm.control.iteration_count),
        static_cast<unsigned long long>(warm.control.skipped_iteration_count),
        static_cast<unsigned long long>(warm.control.conflict_count));
    return reject("telemetry");
  }
  auto saved = plan.snapshot();
  if (!saved)
    return saved.exit_code();
  if (saved->generation() != plan.generation() ||
      saved->fingerprint() != plan.fingerprint() || saved->hash() == 0u) {
    return reject("snapshot");
  }
  evidence.snapshot_hash = saved->hash();
  evidence.pipeline_fingerprint = plan.fingerprint();
  evidence.backend = warm.backend;
  evidence.terminal_reason = second_tick.reason();
  evidence.terminal_code = second_tick.code();
  evidence.generation = plan.generation();
  evidence.command_submits = warm.command_submits;
  evidence.dispatches = warm.dispatches;
  std::array<std::uint32_t, 1u> event_count_value{};
  std::array<std::uint32_t, 1u> integrated_count_value{};
  std::array<std::int32_t, 1u> hash{};
  const Status read_state = plan.read(*pending_state, evidence.state);
  if (!read_state)
    return read_state.exit_code();
  const Status read_events = plan.read(*events, evidence.events);
  if (!read_events)
    return read_events.exit_code();
  const Status read_count = plan.read(*event_count, event_count_value);
  if (!read_count)
    return read_count.exit_code();
  const Status read_bins = plan.read(*resolved, evidence.resolved);
  if (!read_bins)
    return read_bins.exit_code();
  const Status read_hash = plan.read(*state_hash, hash);
  if (!read_hash)
    return read_hash.exit_code();
  const Status read_integrated = plan.read(*integrated, evidence.integrated);
  if (!read_integrated)
    return read_integrated.exit_code();
  const Status read_integrated_count =
      plan.read(*integrated_count, integrated_count_value);
  if (!read_integrated_count)
    return read_integrated_count.exit_code();

  constexpr std::array<std::int32_t, Capacity> expected_state{6, -4, 5,  -2,
                                                              4, -1, -3, 3};
  constexpr std::array<std::int32_t, EventCapacity> expected_events{19, 13};
  constexpr std::array<std::int32_t, EventCapacity> expected_integrated{38, 26};
  constexpr std::array<std::int32_t, IslandCount> expected_resolved{21, 15};
  constexpr std::int32_t expected_state_hash = 1037799664;
  const Stats observed = plan.stats();
  evidence.event_count = event_count_value[0u];
  evidence.state_hash = hash[0u];
  evidence.iterations = warm.control.iteration_count;
  evidence.conflicts = warm.control.conflict_count;
  if (evidence.state != expected_state || evidence.events != expected_events ||
      evidence.integrated != expected_integrated ||
      evidence.resolved != expected_resolved ||
      evidence.state_hash != expected_state_hash ||
      evidence.event_count != EventCapacity ||
      evidence.event_count != integrated_count_value[0u] ||
      observed.pipeline.step_count != 2u ||
      observed.publication.snapshot_byte_count != sizeof(initial) ||
      observed.publication.snapshot_hash != evidence.snapshot_hash) {
    std::fprintf(
        stderr,
        "installed pipeline output counts=%u/%u hash=%d/%d "
        "steps=%llu bytes=%llu snapshots=%llu/%llu\n",
        evidence.event_count, integrated_count_value[0u], evidence.state_hash,
        expected_state_hash,
        static_cast<unsigned long long>(observed.pipeline.step_count),
        static_cast<unsigned long long>(
            observed.publication.snapshot_byte_count),
        static_cast<unsigned long long>(observed.publication.snapshot_hash),
        static_cast<unsigned long long>(evidence.snapshot_hash));
    return reject("output");
  }
  return 0;
}

} // namespace package_device_program
