#pragma once

#include <node/runtime/replay/model.hpp>

#include <cstdint>
#include <string_view>
#include <vector>

namespace rund::node {

namespace replay_detail {

inline constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
inline constexpr std::uint64_t kFnvPrime = 1099511628211ull;

inline void Mix(std::uint64_t &hash, const std::uint64_t value) noexcept {
  hash ^= value;
  hash *= kFnvPrime;
}

inline void MixBool(std::uint64_t &hash, const bool value) noexcept {
  Mix(hash, value ? 1u : 0u);
}

inline void MixSigned(std::uint64_t &hash, const std::int64_t value) noexcept {
  Mix(hash, static_cast<std::uint64_t>(value));
}

inline void MixString(std::uint64_t &hash,
                      const std::string_view value) noexcept {
  Mix(hash, static_cast<std::uint64_t>(value.size()));
  for (const char ch : value) {
    Mix(hash, static_cast<unsigned char>(ch));
  }
}

inline std::uint64_t StartHash(const std::uint64_t checkpoint_hash) noexcept {
  std::uint64_t hash = kFnvOffset;
  MixString(hash, "rund.replay.start");
  Mix(hash, checkpoint_hash);
  return hash;
}

inline std::uint64_t GenesisStartHash() noexcept { return StartHash(0u); }

inline void MixSnapshot(std::uint64_t &hash,
                        const ::rund::TraceRecord::State &snapshot) noexcept {
  Mix(hash, static_cast<std::uint64_t>(snapshot.state));
  Mix(hash, snapshot.active_compute_jobs);
  MixBool(hash, snapshot.scope_active);
  Mix(hash, static_cast<std::uint64_t>(snapshot.code));
}

inline std::uint64_t
HashObservation(const task::Observation &observation) noexcept {
  std::uint64_t hash = kFnvOffset;
  Mix(hash, observation.sequence);
  Mix(hash, static_cast<std::uint64_t>(observation.kind));
  Mix(hash, observation.task_id);
  Mix(hash, observation.wait_id);
  MixSigned(hash, observation.fd);
  Mix(hash, static_cast<std::uint16_t>(observation.interest));
  Mix(hash, static_cast<std::uint16_t>(observation.revents));
  MixSigned(hash, observation.deadline_ns);
  Mix(hash, static_cast<std::uint64_t>(observation.reason_code));
  return hash;
}

inline std::uint64_t
HashObservations(const std::vector<task::Observation> &observations) noexcept {
  std::uint64_t hash = kFnvOffset;
  Mix(hash, observations.size());
  for (const task::Observation &observation : observations) {
    Mix(hash, HashObservation(observation));
  }
  return hash;
}

struct InputTranscriptIdentity {
  std::uint64_t count = 0u;
  std::uint64_t hash = 0u;
};

inline InputTranscriptIdentity HashInputs(
    const ::rund::node::replay_detail::payload::Archive &archive) noexcept {
  InputTranscriptIdentity identity{};
  std::uint64_t hash = kFnvOffset;
  for (const ::rund::node::replay_detail::payload::ArchiveRecord &record :
       archive.records) {
    const payload::Record &metadata = record.metadata;
    if (metadata.role != ::rund::node::replay_detail::payload::Role::Input) {
      continue;
    }
    Mix(hash, metadata.input_source);
    Mix(hash, metadata.input_schema);
    Mix(hash, metadata.input_sequence);
    Mix(hash, metadata.source_event_offset);
    Mix(hash, metadata.source_event_count);
    Mix(hash, metadata.source_payload_offset);
    Mix(hash, metadata.source_payload_count);
    Mix(hash, metadata.source_hash);
    Mix(hash, metadata.completed_bytes);
    Mix(hash, metadata.payload_hash.value);
    ++identity.count;
  }
  if (identity.count != 0u) {
    Mix(hash, identity.count);
    identity.hash = hash;
  }
  return identity;
}

inline std::uint64_t HashTrace(const ::rund::Trace &trace) noexcept {
  std::uint64_t hash = kFnvOffset;
  Mix(hash, static_cast<std::uint64_t>(trace.code));
  Mix(hash, trace.dropped);
  Mix(hash, trace.records.size());
  for (const ::rund::TraceRecord &record : trace.records) {
    Mix(hash, static_cast<std::uint64_t>(record.event));
    Mix(hash, static_cast<std::uint64_t>(record.code.domain()));
    Mix(hash, record.code.value());
    MixSnapshot(hash, record.snapshot);
    Mix(hash, record.sequence);
  }
  return hash;
}

inline std::uint64_t
HashTraceRecord(const ::rund::TraceRecord &record) noexcept {
  std::uint64_t hash = kFnvOffset;
  Mix(hash, static_cast<std::uint64_t>(record.event));
  Mix(hash, static_cast<std::uint64_t>(record.code.domain()));
  Mix(hash, record.code.value());
  MixSnapshot(hash, record.snapshot);
  Mix(hash, record.sequence);
  return hash;
}

void MixSchedulerTaskSemantics(std::uint64_t &hash,
                               const task::Stats &tasks) noexcept;

inline std::uint64_t
HashSemantic(const ::rund::replay::Code code, const task::Stats &tasks,
             const std::uint64_t observation_hash) noexcept {
  std::uint64_t hash = kFnvOffset;
  Mix(hash, ::rund::replay::raw(code));
  MixSchedulerTaskSemantics(hash, tasks);
  Mix(hash, observation_hash);
  return hash;
}

inline std::uint64_t HashReplaySubstitutableOperation(
    const task::Stats &tasks, const std::uint64_t observation_hash) noexcept {
  std::uint64_t hash = kFnvOffset;
  MixSchedulerTaskSemantics(hash, tasks);
  Mix(hash, observation_hash);
  return hash;
}

inline std::uint64_t HashReplay(const RuntimeReplayRecord &record) noexcept {
  std::uint64_t hash = kFnvOffset;
  Mix(hash, record.start_hash);
  Mix(hash, record.semantic_hash);
  Mix(hash, record.operation_hash);
  Mix(hash, record.observation_hash);
  Mix(hash, record.host_event_hash);
  Mix(hash, record.input_count);
  Mix(hash, record.input_hash);
  Mix(hash, record.transcript_hash);
  Mix(hash, record.trace_hash);
  return hash;
}

} // namespace replay_detail

} // namespace rund::node
