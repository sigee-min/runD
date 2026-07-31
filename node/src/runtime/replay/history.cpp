#include <rund/replay/history.hpp>

#include <rund/counter.hpp>
#include <kernel/core/checked.hpp>

#include <limits>
#include <utility>

namespace rund::replay {
namespace {

[[nodiscard]] bool ToU64(const std::size_t value,
                         std::uint64_t &converted) noexcept {
  if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
    if (value >
        static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())) {
      return false;
    }
  }
  converted = static_cast<std::uint64_t>(value);
  return true;
}

[[nodiscard]] bool SegmentWeight(const Record &record,
                                 const std::span<const std::byte> state,
                                 std::uint64_t &bytes,
                                 std::uint64_t &events) noexcept {
  const ::rund::replay::StorageReport storage = record.storage_report();
  const ::rund::replay::DiagnosticReport diagnostic = record.capture_report();
  std::uint64_t state_bytes = 0u;
  if (!ToU64(state.size(), state_bytes) ||
      !rund::kernel::checked::add(storage.logical_bytes,
                                        diagnostic.retained_bytes, bytes) ||
      !rund::kernel::checked::add(bytes, state_bytes, bytes)) {
    return false;
  }

  const std::size_t counts[]{record.observation_count(),
                             record.host_event_count(), record.input_count(),
                             record.captures().size(),
                             record.trace_record_count()};
  events = 0u;
  for (const std::size_t count : counts) {
    std::uint64_t converted = 0u;
    if (!ToU64(count, converted) ||
        !rund::kernel::checked::add(events, converted, events)) {
      return false;
    }
  }
  return true;
}

} // namespace

History::History(const Retention retention) noexcept : retention_(retention) {
  std::uint64_t max_segments = 0u;
  if (retention_.max_segments == 0u || retention_.max_bytes == 0u ||
      retention_.max_events == 0u ||
      (ToU64(slots_.max_size(), max_segments) &&
       retention_.max_segments > max_segments)) {
    code_ = ::rund::replay::Code::RetentionInvalid;
    return;
  }
  try {
    slots_.resize(retention_.max_segments);
    code_ = Code::Ok;
  } catch (...) {
    slots_.clear();
    code_ = ::rund::replay::Code::RetentionAllocationFailed;
  }
}

History::History(Checkpoint previous, const Retention retention) noexcept
    : History(retention) {
  if (!ok()) {
    return;
  }
  if (!previous) {
    slots_.clear();
    code_ = previous.code();
    return;
  }
  if (previous.state_size() > retention_.max_bytes) {
    slots_.clear();
    code_ = ::rund::replay::Code::RetentionAnchorExceedsBounds;
    return;
  }
  retained_bytes_ = static_cast<std::uint64_t>(previous.state_size());
  anchor_.emplace(std::move(previous));
}

History::History(History &&other) noexcept
    : retention_(other.retention_), anchor_(std::move(other.anchor_)),
      slots_(std::move(other.slots_)), head_(other.head_), size_(other.size_),
      retained_bytes_(other.retained_bytes_),
      retained_events_(other.retained_events_),
      appended_segments_(other.appended_segments_),
      evicted_segments_(other.evicted_segments_),
      rejected_segments_(other.rejected_segments_), code_(other.code_) {
  other.reset_moved_from();
}

History &History::operator=(History &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  retention_ = other.retention_;
  anchor_ = std::move(other.anchor_);
  slots_ = std::move(other.slots_);
  head_ = other.head_;
  size_ = other.size_;
  retained_bytes_ = other.retained_bytes_;
  retained_events_ = other.retained_events_;
  appended_segments_ = other.appended_segments_;
  evicted_segments_ = other.evicted_segments_;
  rejected_segments_ = other.rejected_segments_;
  code_ = other.code_;
  other.reset_moved_from();
  return *this;
}

void History::reset_moved_from() noexcept {
  anchor_.reset();
  slots_.clear();
  head_ = 0u;
  size_ = 0u;
  retained_bytes_ = 0u;
  retained_events_ = 0u;
  appended_segments_ = 0u;
  evicted_segments_ = 0u;
  rejected_segments_ = 0u;
  code_ = ::rund::replay::Code::HistoryMovedFrom;
}

Report History::report() const noexcept {
  Report value{
      .retained_segments = size_,
      .retained_bytes = retained_bytes_,
      .retained_events = retained_events_,
      .appended_segments = appended_segments_,
      .evicted_segments = evicted_segments_,
      .rejected_segments = rejected_segments_,
  };
  if (size_ == 0u || slots_.empty()) {
    return value;
  }
  const Segment &oldest = *slots_[head_];
  const std::size_t latest_index = (head_ + size_ - 1u) % slots_.size();
  const Segment &latest = *slots_[latest_index];
  value.oldest_sequence = oldest.sequence();
  value.newest_sequence = latest.sequence();
  value.prefix_hash = latest.checkpoint().prefix_hash();
  value.transcript_prefix_hash = latest.checkpoint().transcript_prefix_hash();
  return value;
}

std::optional<Segment>
History::segment(const std::size_t index) const noexcept {
  if (index >= size_ || slots_.empty()) {
    return std::nullopt;
  }
  return slots_[(head_ + index) % slots_.size()];
}

std::optional<Segment>
History::find(const std::uint64_t sequence) const noexcept {
  if (size_ == 0u || slots_.empty() || sequence == 0u) {
    return std::nullopt;
  }
  const std::optional<Segment> &oldest = slots_[head_];
  if (!oldest.has_value() || sequence < oldest->sequence()) {
    return std::nullopt;
  }
  const std::uint64_t offset = sequence - oldest->sequence();
  std::uint64_t count = 0u;
  if (!ToU64(size_, count) || offset >= count) {
    return std::nullopt;
  }
  if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t)) {
    if (offset > std::numeric_limits<std::size_t>::max()) {
      return std::nullopt;
    }
  }
  const std::size_t index = static_cast<std::size_t>(offset);
  const std::size_t until_wrap = slots_.size() - head_;
  const std::size_t slot =
      index < until_wrap ? head_ + index : index - until_wrap;
  return slots_[slot].has_value() && slots_[slot]->sequence() == sequence
             ? slots_[slot]
             : std::nullopt;
}

Append History::append(const Binding &binding, Record record,
                       const std::span<const std::byte> state) noexcept {
  const auto reject = [this](const Code code) noexcept {
    ::rund::detail::counter::Accumulate(rejected_segments_, 1u);
    return Append{code, 0u, 0u};
  };
  if (!ok()) {
    return reject(code_);
  }
  if (!record) {
    return reject(::rund::replay::Code::RetentionRecordInvalid);
  }
  if (!binding) {
    return reject(binding.code());
  }
  if (!binding.checkpointable()) {
    return reject(::rund::replay::Code::StateSchemaInvalid);
  }

  std::uint64_t segment_bytes = 0u;
  std::uint64_t segment_events = 0u;
  if (!SegmentWeight(record, state, segment_bytes, segment_events)) {
    return reject(::rund::replay::Code::RetentionWeightOverflow);
  }
  if (segment_bytes > retention_.max_bytes ||
      segment_events > retention_.max_events) {
    return reject(::rund::replay::Code::RetentionSegmentExceedsBounds);
  }

  std::size_t remove = 0u;
  std::uint64_t remaining_bytes =
      size_ == 0u && anchor_.has_value() ? 0u : retained_bytes_;
  std::uint64_t remaining_events = retained_events_;
  if (remaining_bytes > retention_.max_bytes ||
      remaining_events > retention_.max_events) {
    return reject(::rund::replay::Code::RetentionStateInvalid);
  }
  while (size_ - remove >= retention_.max_segments ||
         segment_bytes > retention_.max_bytes - remaining_bytes ||
         segment_events > retention_.max_events - remaining_events) {
    const std::size_t index = (head_ + remove) % slots_.size();
    if (!slots_[index].has_value()) {
      return reject(::rund::replay::Code::RetentionStateInvalid);
    }
    const Segment &evicted = *slots_[index];
    if (evicted.byte_count() > remaining_bytes ||
        evicted.event_count() > remaining_events) {
      return reject(::rund::replay::Code::RetentionStateInvalid);
    }
    remaining_bytes -= evicted.byte_count();
    remaining_events -= evicted.event_count();
    ++remove;
  }

  std::optional<Checkpoint> next{};
  try {
    if (size_ != 0u) {
      const std::size_t latest = (head_ + size_ - 1u) % slots_.size();
      next.emplace(
          binding.advance(slots_[latest]->checkpoint(), record, state));
    } else if (anchor_.has_value()) {
      next.emplace(binding.advance(*anchor_, record, state));
    } else {
      next.emplace(binding.checkpoint(record, state));
    }
  } catch (...) {
    return reject(::rund::replay::Code::RetentionAllocationFailed);
  }
  if (!*next) {
    return reject(next->code());
  }
  const std::uint64_t sequence = next->segment_count();

  for (std::size_t index = 0u; index < remove; ++index) {
    slots_[head_].reset();
    head_ = (head_ + 1u) % slots_.size();
    --size_;
  }
  retained_bytes_ = remaining_bytes;
  retained_events_ = remaining_events;
  ::rund::detail::counter::Accumulate(evicted_segments_, remove);

  const std::size_t tail = (head_ + size_) % slots_.size();
  Segment segment{sequence, segment_bytes, segment_events, std::move(record),
                  std::move(*next)};
  slots_[tail].emplace(std::move(segment));
  ++size_;
  retained_bytes_ += segment_bytes;
  retained_events_ += segment_events;
  ::rund::detail::counter::Accumulate(appended_segments_, 1u);
  anchor_.reset();
  return Append{Code::Ok, sequence, remove};
}

Append Binding::append(History &history, Record record,
                       const std::span<const std::byte> state) const noexcept {
  return history.append(*this, std::move(record), state);
}

} // namespace rund::replay
