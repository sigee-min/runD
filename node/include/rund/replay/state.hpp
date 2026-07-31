#pragma once

#include <rund/replay/record.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

namespace rund::replay {

class Scenario;

// A checkpoint is a quiescent boundary between completed replay segments. It
// copies domain-neutral caller bytes once into a private immutable snapshot and
// owns only that snapshot plus canonical identities; it never captures a live
// Runtime, coroutine, task, or device resource.
class Checkpoint final {
public:
  Checkpoint(const Checkpoint &) noexcept = default;
  Checkpoint &operator=(const Checkpoint &) noexcept = default;
  Checkpoint(Checkpoint &&) noexcept = default;
  Checkpoint &operator=(Checkpoint &&) noexcept = default;

  [[nodiscard]] bool ok() const noexcept;
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
  [[nodiscard]] Code code() const noexcept;
  [[nodiscard]] std::string_view error() const noexcept;
  [[nodiscard]] int exit_code() const noexcept {
    return ::rund::replay::exit_code(code());
  }
  [[nodiscard]] std::uint64_t segment_count() const noexcept;
  [[nodiscard]] std::uint64_t input_position() const noexcept;
  [[nodiscard]] std::uint64_t schema() const noexcept;
  [[nodiscard]] std::size_t state_size() const noexcept;
  [[nodiscard]] std::span<const std::byte> state() const noexcept;
  [[nodiscard]] std::uint64_t state_hash() const noexcept;
  [[nodiscard]] std::uint64_t boundary_hash() const noexcept;
  [[nodiscard]] std::uint64_t prefix_hash() const noexcept;
  [[nodiscard]] std::uint64_t transcript_prefix_hash() const noexcept;
  [[nodiscard]] std::uint64_t hash() const noexcept;
  template <typename Sink>
    requires std::invocable<Sink &, std::span<const std::byte>> &&
             std::convertible_to<
                 std::invoke_result_t<Sink &, std::span<const std::byte>>, bool>
  [[nodiscard]] Save save(Sink &&sink) const noexcept {
    return persist(detail::output(sink));
  }
  [[nodiscard]] static Load<Checkpoint>
  load(std::span<const std::byte> artifact, Limits limits = {}) noexcept;

private:
  struct Data;

  explicit Checkpoint(std::shared_ptr<const Data> data) noexcept
      : data_(std::move(data)) {}
  explicit Checkpoint(const Code code) noexcept : code_(code) {}
  [[nodiscard]] Save persist(detail::Output output) const noexcept;

  std::shared_ptr<const Data> data_{};
  Code code_ = Code::CheckpointMovedFrom;

  friend class Binding;
  friend class Load<Checkpoint>;
};

// A resume binding owns the immutable checkpoint and borrows one restore
// callback. Schema is validated once when the binding is made;
// every execution then enters the same detail::Access authority.
class Resume final {
public:
  Resume(const Resume &) = delete;
  Resume &operator=(const Resume &) = delete;
  Resume(Resume &&) = delete;
  Resume &operator=(Resume &&) = delete;

  [[nodiscard]] constexpr bool ok() const noexcept { return code_ == Code::Ok; }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
  [[nodiscard]] constexpr Code code() const noexcept { return code_; }
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::replay::error(code_);
  }
  [[nodiscard]] constexpr int exit_code() const noexcept {
    return ::rund::replay::exit_code(code_);
  }

  template <typename Callback>
  [[nodiscard]] Record record(Session &session, Callback &&callback) const;
  template <typename Callback>
  [[nodiscard]] Check run(Session &session, const Record &expected,
                          Callback &&callback) const;
  template <typename Callback>
  [[nodiscard]] Scenario scenario(Session &session, const Record &expected,
                                  std::span<const Choice> choices,
                                  Callback &&callback) const;

private:
  Resume(Checkpoint checkpoint, const detail::RestoreCall restore,
         const Code code) noexcept
      : checkpoint_(std::move(checkpoint)), restore_(restore), code_(code) {}

  Checkpoint checkpoint_;
  detail::RestoreCall restore_{};
  Code code_ = Code::CheckpointRestoreInvalid;

  friend class Binding;
  friend struct detail::Access;
};

template <typename T> class Load final {
  static_assert(std::same_as<T, Record> || std::same_as<T, Checkpoint>,
                "replay Load supports Record and Checkpoint");

public:
  Load(const Load &) noexcept = default;
  Load &operator=(const Load &) noexcept = default;
  Load(Load &&other) noexcept
      : code_(other.code_), value_(std::move(other.value_)) {
    other.reset_moved_from();
  }
  Load &operator=(Load &&other) noexcept {
    if (this != &other) {
      code_ = other.code_;
      value_ = std::move(other.value_);
      other.reset_moved_from();
    }
    return *this;
  }

  [[nodiscard]] bool ok() const noexcept { return code_ == Code::Ok; }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
  [[nodiscard]] Code code() const noexcept { return code_; }
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::replay::error(code_);
  }
  [[nodiscard]] int exit_code() const noexcept {
    return ::rund::replay::exit_code(code());
  }
  [[nodiscard]] const T &operator*() const noexcept { return *value_; }
  [[nodiscard]] const T *operator->() const noexcept {
    return std::addressof(*value_);
  }

private:
  [[nodiscard]] static constexpr Code empty_code() noexcept {
    if constexpr (std::same_as<T, Record>) {
      return Code::RecordNotLoaded;
    } else {
      return Code::CheckpointNotLoaded;
    }
  }

  [[nodiscard]] static constexpr Code moved_code() noexcept {
    if constexpr (std::same_as<T, Record>) {
      return Code::RecordLoadMovedFrom;
    } else {
      return Code::CheckpointLoadMovedFrom;
    }
  }

  Load(const Code code, std::optional<T> value) noexcept
      : code_(code), value_(std::move(value)) {
    if (value_ && value_->ok()) {
      code_ = Code::Ok;
    } else {
      value_.reset();
      if (code_ == Code::Ok) {
        code_ = empty_code();
      }
    }
  }

  void reset_moved_from() noexcept {
    code_ = moved_code();
    value_.reset();
  }

  Code code_ = empty_code();
  std::optional<T> value_{};

  friend class Record;
  friend class Checkpoint;
};

} // namespace rund::replay
