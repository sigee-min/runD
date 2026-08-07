#include "local.hpp"

#include "../exception.hpp"
#include "../host/codec.hpp"
#include "../host/payload/store.hpp"

#include <node/runtime/replay/codec/decode.hpp>
#include <node/runtime/replay/hash.hpp>
#include <node/runtime/replay/host/payload.hpp>
#include <node/runtime/replay/record.hpp>
#include <node/runtime/replay/task/stats.hpp>

#include "../../task/stats/access.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace rund::node {

RuntimeReplayDecodeResult
DecodeRuntimeReplayRecord(const std::span<const std::byte> encoded,
                          const ::rund::replay::Limits limits) {
  if (encoded.size() > limits.max_bytes) {
    return RuntimeReplayDecodeResult{
        .code = ::rund::replay::Code::CodecEncodedCapacityExceeded};
  }
  try {
    replay_detail::artifact::Reader in{encoded};
    if (!in.header(replay_detail::artifact::Kind::Record)) {
      return RuntimeReplayDecodeResult{
          .code = ::rund::replay::Code::CodecBadHeader};
    }
    std::uint64_t raw_code = 0u;
    std::uint64_t start_hash = 0u;
    std::uint64_t expected_hash = 0u;
    ::rund::replay::Code code{};
    if (!in.varuint(raw_code) ||
        !replay_detail::artifact::runtime_code(raw_code, code) ||
        !in.fixed64(start_hash) || start_hash == 0u ||
        !in.fixed64(expected_hash)) {
      return RuntimeReplayDecodeResult{.code =
                                           ::rund::replay::Code::CodecBadField};
    }

    task::Stats tasks{};
    for (const replay_detail::SchedulerStatSlot slot :
         replay_detail::kSchedulerReplaySlots) {
      if (!in.varuint(
              ::rund::detail::task::StatsAccess::Counter(tasks, slot))) {
        return RuntimeReplayDecodeResult{
            .code = ::rund::replay::Code::CodecBadField};
      }
    }

    replay_detail::artifact::Admission admission{limits};
    std::vector<task::Observation> observations{};
    if (!replay_detail::artifact::read_observations(in, admission,
                                                    observations)) {
      return RuntimeReplayDecodeResult{
          .code = admission.failure(::rund::replay::Code::CodecBadField)};
    }
    replay_detail::HostReplayDecodeResult host =
        replay_detail::ReadHostEvidence(in, limits.max_entries);
    if (!host.ok() ||
        !admission.entries(static_cast<std::uint64_t>(host.events.size()))) {
      return RuntimeReplayDecodeResult{
          .code = host.ok() ? ::rund::replay::Code::CodecEntryCapacityExceeded
                            : host.code};
    }
    replay_detail::payload::Archive archive{};
    if (!replay_detail::artifact::read_payload(in, admission, host.events,
                                               archive)) {
      return RuntimeReplayDecodeResult{
          .code = admission.failure(::rund::replay::Code::CodecBadField)};
    }
    ::rund::Trace trace{};
    if (!replay_detail::artifact::read_trace(in, admission, trace) ||
        !in.done()) {
      return RuntimeReplayDecodeResult{
          .code = admission.failure(::rund::replay::Code::CodecBadField)};
    }

    RuntimeReplayRecord record = make_runtime_replay_record(
        RuntimeReplayRecordDesc{.start_hash = start_hash,
                                .code = code,
                                .tasks = tasks,
                                .observations = std::move(observations),
                                .host_events = std::move(host.events),
                                .host_payload_archive = std::move(archive),
                                .trace = std::move(trace)});
    const ::rund::replay::Code payload_code = replay_detail::BindPayloads(
        record.host.events, record.host.payload_archive);
    if (payload_code != ::rund::replay::Code::Ok) {
      return RuntimeReplayDecodeResult{.code = payload_code};
    }
    if (record.replay_hash != expected_hash) {
      return RuntimeReplayDecodeResult{
          .code = ::rund::replay::Code::CodecHashInvalid};
    }
    if (!valid_runtime_replay_record(record)) {
      return RuntimeReplayDecodeResult{
          .code = ::rund::replay::Code::CodecInvariantInvalid};
    }
    return RuntimeReplayDecodeResult{.code = ::rund::replay::Code::Ok,
                                     .record = std::move(record)};
  } catch (...) {
    return RuntimeReplayDecodeResult{
        .code = replay_detail::CurrentExceptionCode({
            .bad_alloc = ::rund::replay::Code::CodecCapacityExceeded,
            .length_error = ::rund::replay::Code::CodecCapacityExceeded,
            .unexpected = ::rund::replay::Code::CodecLoadFailed,
        })};
  }
}

} // namespace rund::node
