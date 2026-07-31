#include "data.hpp"

#include "../../session/result.hpp"
#include "../host/payload/store.hpp"
#include "../record/factory.hpp"

#include <node/runtime/replay/code.hpp>
#include <node/runtime/replay/codec.hpp>
#include <rund/replay/state.hpp>

#include <memory>
#include <utility>

namespace rund::replay {
namespace {

[[nodiscard]] node::RuntimeReplayRecord
consume(Session::Result &result, const std::uint64_t start_hash) {
  return node::make_runtime_replay_record(node::RuntimeReplayRecordDesc{
      .start_hash = start_hash,
      .code = node::replay_detail::code(result.code()),
      .tasks = result.tasks(),
      .observations =
          ::rund::detail::session::ResultAccess::take_observations(result),
      .host_events = ::rund::detail::session::ResultAccess::take_events(result),
      .host_payload_archive =
          ::rund::detail::session::ResultAccess::take_payloads(result),
      .trace = ::rund::detail::session::ResultAccess::take_trace(result),
  });
}

[[nodiscard]] bool project(const node::RuntimeReplayRecord &record,
                           std::vector<Capture> &captures) {
  const node::replay_detail::payload::DiagnosticArchive &archive =
      record.host.payload_archive.diagnostic;
  const std::span<const std::byte> bytes = archive.bytes.span();
  captures.reserve(archive.records.size());
  for (const node::replay_detail::payload::DiagnosticRecord &source :
       archive.records) {
    if (source.offset > bytes.size() ||
        source.byte_count > bytes.size() - source.offset) {
      captures.clear();
      return false;
    }
    captures.push_back(Capture{
        .sequence = source.event_sequence,
        .kind = source.kind,
        .bytes = bytes.subspan(static_cast<std::size_t>(source.offset),
                               static_cast<std::size_t>(source.byte_count)),
        .hash = source.payload_hash.value});
  }
  return true;
}

} // namespace

Record::Data::Data(Session &session, Session::Result &&result,
                   const std::uint64_t start_hash)
    : record(consume(result, start_hash)) {
  const Code result_code =
      result.ok() && result.tasks().host_events_dropped() != 0u
          ? Code::HostEventRetentionExhausted
          : Code::Ok;
  detail::scope::Prepared ready = detail::scope::Access::prepare(
      session, record.host.events, record.host.payload_archive);
  prepared = std::move(ready.owner);
  published.store(prepared.get(), std::memory_order_release);
  facade_code = prepared ? result_code : ready.code;
  if (!project(record, captures)) {
    facade_code = Code::HostDiagnosticHashInvalid;
  }
}

Record::Data::Data(node::RuntimeReplayRecord &&value)
    : record(std::move(value)) {
  if (!project(record, captures)) {
    facade_code = Code::HostDiagnosticHashInvalid;
  }
}

namespace detail {

Record build_record(Session &session, Session::Result &&result,
                    const std::uint64_t start_hash) noexcept {
  try {
    return Record{std::make_shared<const Record::Data>(
        session, std::move(result), start_hash)};
  } catch (...) {
    return Record{Code::AllocationFailed};
  }
}

Record fail_record(const Code code) noexcept {
  return Record{code == Code::Ok ? Code::CheckpointInvalid : code};
}

Code ready_code(const Record &record) noexcept {
  return record.data_ ? record.data_->facade_code : record.code_;
}

} // namespace detail

bool Record::ok() const noexcept { return code() == Code::Ok; }

Code Record::code() const noexcept {
  if (!data_) {
    return code_;
  }
  return data_->facade_code == Code::Ok ? data_->record.code
                                        : data_->facade_code;
}

std::string_view Record::error() const noexcept {
  return ::rund::replay::error(code());
}

int Record::exit_code() const noexcept {
  return ::rund::replay::exit_code(code());
}

const task::Stats &Record::tasks() const noexcept {
  static constexpr task::Stats empty{};
  return data_ ? data_->record.tasks : empty;
}

std::size_t Record::observation_count() const noexcept {
  return data_ && data_->facade_code == Code::Ok
             ? data_->record.observations.size()
             : 0u;
}

std::size_t Record::host_event_count() const noexcept {
  return data_ && data_->facade_code == Code::Ok
             ? data_->record.host.events.size()
             : 0u;
}

std::size_t Record::input_count() const noexcept {
  return data_ && data_->facade_code == Code::Ok
             ? static_cast<std::size_t>(data_->record.input_count)
             : 0u;
}

std::uint64_t Record::input_hash() const noexcept {
  return data_ && data_->facade_code == Code::Ok ? data_->record.input_hash
                                                 : 0u;
}

std::span<const Capture> Record::captures() const noexcept {
  return data_ && data_->facade_code == Code::Ok
             ? std::span<const Capture>{data_->captures}
             : std::span<const Capture>{};
}

std::uint64_t Record::capture_hash() const noexcept {
  return data_ && data_->facade_code == Code::Ok
             ? data_->record.host.payload_archive.diagnostic.hash
             : 0u;
}

DiagnosticReport Record::capture_report() const noexcept {
  return data_ && data_->facade_code == Code::Ok
             ? data_->record.host.payload_archive.diagnostic.report
             : DiagnosticReport{};
}

StorageReport Record::storage_report() const noexcept {
  return data_ ? data_->record.host.payload_archive.storage : StorageReport{};
}

std::size_t Record::trace_record_count() const noexcept {
  return data_ && data_->facade_code == Code::Ok
             ? data_->record.trace.records.size()
             : 0u;
}

std::uint64_t Record::semantic_hash() const noexcept {
  return data_ && data_->facade_code == Code::Ok ? data_->record.semantic_hash
                                                 : 0u;
}

std::uint64_t Record::operation_hash() const noexcept {
  return data_ && data_->facade_code == Code::Ok ? data_->record.operation_hash
                                                 : 0u;
}

std::uint64_t Record::observation_hash() const noexcept {
  return data_ && data_->facade_code == Code::Ok
             ? data_->record.observation_hash
             : 0u;
}

std::uint64_t Record::host_event_hash() const noexcept {
  return data_ && data_->facade_code == Code::Ok ? data_->record.host_event_hash
                                                 : 0u;
}

std::uint64_t Record::transcript_hash() const noexcept {
  return data_ && data_->facade_code == Code::Ok ? data_->record.transcript_hash
                                                 : 0u;
}

std::uint64_t Record::trace_hash() const noexcept {
  return data_ && data_->facade_code == Code::Ok ? data_->record.trace_hash
                                                 : 0u;
}

std::uint64_t Record::start_hash() const noexcept {
  return data_ && data_->facade_code == Code::Ok ? data_->record.start_hash
                                                 : 0u;
}

std::uint64_t Record::hash() const noexcept {
  return data_ && data_->facade_code == Code::Ok ? data_->record.replay_hash
                                                 : 0u;
}

Save Record::persist(const detail::Output output) const noexcept {
  const Code code = detail::ready_code(*this);
  if (code != Code::Ok) {
    return Save{code, 0u, 0u};
  }
  const node::replay_detail::payload::Store *payloads =
      data_->prepared == nullptr ? nullptr : &data_->prepared->payloads();
  const node::replay_detail::artifact::Result saved =
      node::replay_detail::artifact::save(
          data_->record, payloads,
          node::replay_detail::artifact::Sink{.state = output.state,
                                              .write = output.write});
  return Save{saved.code, saved.bytes, saved.writes};
}

Load<Record> Record::load(const std::span<const std::byte> artifact,
                          const Limits limits) noexcept {
  try {
    node::RuntimeReplayDecodeResult decoded =
        node::DecodeRuntimeReplayRecord(artifact, limits);
    if (!decoded.ok()) {
      return Load<Record>{decoded.code, std::nullopt};
    }
    if (decoded.record.tasks.host_events_dropped() != 0u) {
      return Load<Record>{Code::HostEventRetentionExhausted, std::nullopt};
    }
    return Load<Record>{Code::Ok, Record{std::make_shared<const Record::Data>(
                                      std::move(decoded.record))}};
  } catch (...) {
    return Load<Record>{Code::CodecCapacityExceeded, std::nullopt};
  }
}

} // namespace rund::replay
