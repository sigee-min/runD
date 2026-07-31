#pragma once

#include <rund/replay/state.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace rund::replay {

struct Retention final {
  std::size_t max_segments = 64u;
  std::uint64_t max_bytes = 1024u * 1024u * 1024u;
  std::uint64_t max_events = 1024u * 1024u;
};

struct Report final {
  std::size_t retained_segments = 0u;
  std::uint64_t retained_bytes = 0u;
  std::uint64_t retained_events = 0u;
  std::uint64_t appended_segments = 0u;
  std::uint64_t evicted_segments = 0u;
  std::uint64_t rejected_segments = 0u;
  std::uint64_t oldest_sequence = 0u;
  std::uint64_t newest_sequence = 0u;
  std::uint64_t prefix_hash = 0u;
  std::uint64_t transcript_prefix_hash = 0u;
};

class Segment final {
public:
  Segment(const Segment &) noexcept = default;
  Segment &operator=(const Segment &) noexcept = default;
  Segment(Segment &&) noexcept = default;
  Segment &operator=(Segment &&) noexcept = default;

  [[nodiscard]] std::uint64_t sequence() const noexcept { return sequence_; }
  [[nodiscard]] std::uint64_t byte_count() const noexcept {
    return byte_count_;
  }
  [[nodiscard]] std::uint64_t event_count() const noexcept {
    return event_count_;
  }
  [[nodiscard]] const Record &record() const noexcept { return record_; }
  [[nodiscard]] const Checkpoint &checkpoint() const noexcept {
    return checkpoint_;
  }

private:
  Segment(std::uint64_t sequence, std::uint64_t byte_count,
          std::uint64_t event_count, Record record,
          Checkpoint checkpoint) noexcept
      : sequence_(sequence), byte_count_(byte_count), event_count_(event_count),
        record_(std::move(record)), checkpoint_(std::move(checkpoint)) {}

  std::uint64_t sequence_ = 0u;
  std::uint64_t byte_count_ = 0u;
  std::uint64_t event_count_ = 0u;
  Record record_;
  Checkpoint checkpoint_;

  friend class History;
};

class Append final {
public:
  [[nodiscard]] constexpr bool ok() const noexcept { return code_ == Code::Ok; }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
  [[nodiscard]] constexpr Code code() const noexcept { return code_; }
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::replay::error(code_);
  }
  [[nodiscard]] constexpr int exit_code() const noexcept {
    return ::rund::replay::exit_code(code());
  }
  [[nodiscard]] constexpr std::uint64_t sequence() const noexcept {
    return sequence_;
  }
  [[nodiscard]] constexpr std::size_t evicted_segments() const noexcept {
    return evicted_segments_;
  }

private:
  constexpr Append(Code code, std::uint64_t sequence,
                   std::size_t evicted_segments) noexcept
      : code_(code), sequence_(sequence), evicted_segments_(evicted_segments) {}

  Code code_ = Code::SegmentNotAppended;
  std::uint64_t sequence_ = 0u;
  std::size_t evicted_segments_ = 0u;

  friend class History;
};

class History final {
public:
  explicit History(Retention retention = {}) noexcept;
  History(Checkpoint previous, Retention retention = {}) noexcept;
  History(const History &) = delete;
  History &operator=(const History &) = delete;
  History(History &&other) noexcept;
  History &operator=(History &&other) noexcept;

  [[nodiscard]] bool ok() const noexcept { return code_ == Code::Ok; }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
  [[nodiscard]] Code code() const noexcept { return code_; }
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::replay::error(code_);
  }
  [[nodiscard]] int exit_code() const noexcept {
    return ::rund::replay::exit_code(code());
  }
  [[nodiscard]] const Retention &retention() const noexcept {
    return retention_;
  }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0u; }
  [[nodiscard]] Report report() const noexcept;
  [[nodiscard]] std::optional<Segment>
  segment(std::size_t index) const noexcept;
  [[nodiscard]] std::optional<Segment>
  find(std::uint64_t sequence) const noexcept;
  [[nodiscard]] std::optional<Segment> oldest() const noexcept {
    return segment(0u);
  }
  [[nodiscard]] std::optional<Segment> latest() const noexcept {
    return size_ == 0u ? std::nullopt : segment(size_ - 1u);
  }

private:
  [[nodiscard]] Append append(const Binding &binding, Record record,
                              std::span<const std::byte> state) noexcept;
  void reset_moved_from() noexcept;

  Retention retention_{};
  std::optional<Checkpoint> anchor_{};
  std::vector<std::optional<Segment>> slots_{};
  std::size_t head_ = 0u;
  std::size_t size_ = 0u;
  std::uint64_t retained_bytes_ = 0u;
  std::uint64_t retained_events_ = 0u;
  std::uint64_t appended_segments_ = 0u;
  std::uint64_t evicted_segments_ = 0u;
  std::uint64_t rejected_segments_ = 0u;
  Code code_ = Code::HistoryNotPrepared;

  friend class Binding;
};

} // namespace rund::replay
