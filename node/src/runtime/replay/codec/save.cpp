#include "local.hpp"

#include "../host/codec.hpp"
#include "../host/payload/hash.hpp"

#include <node/runtime/replay/codec/save.hpp>
#include <node/runtime/replay/task/stats.hpp>

#include "../../task/stats/access.hpp"

namespace rund::node::replay_detail::artifact {

Result save(const RuntimeReplayRecord &record,
            const payload::Store *const payloads, const Sink sink) noexcept {
  Writer out{sink};
  ::rund::replay::Code stored_code = ::rund::replay::Code::Ok;
  static_cast<void>(
      out.require(runtime_code(::rund::replay::raw(record.code), stored_code) &&
                      stored_code == record.code && record.start_hash != 0u,
                  ::rund::replay::Code::CodecInvariantInvalid));
  static_cast<void>(out.header(Kind::Record) &&
                    out.varuint(::rund::replay::raw(record.code)) &&
                    out.fixed64(record.start_hash) &&
                    out.fixed64(record.replay_hash));
  for (const replay_detail::SchedulerStatSlot slot :
       replay_detail::kSchedulerReplaySlots) {
    static_cast<void>(out.varuint(
        ::rund::detail::task::StatsAccess::Value(record.tasks, slot)));
  }
  static_cast<void>(out.require(write_observations(out, record.observations),
                                ::rund::replay::Code::CodecInvariantInvalid));
  static_cast<void>(
      out.require(replay_detail::WriteHostEvidence(out, record.host.events,
                                                   record.host.event_hash),
                  ::rund::replay::Code::CodecInvariantInvalid));
  static_cast<void>(out.require(
      write_payload(out, record.host.payload_archive, payloads,
                    replay_detail::EventsRequirePayload(record.host.events)),
      ::rund::replay::Code::CodecInvariantInvalid));
  static_cast<void>(out.require(write_trace(out, record.trace),
                                ::rund::replay::Code::CodecInvariantInvalid));
  return out.finish();
}

} // namespace rund::node::replay_detail::artifact
