#pragma once

#include <rund/task/api.hpp>
#include <rund/task/await.hpp>
#include <rund/task/coroutine.hpp>
#include <rund/task/status.hpp>

#include <cstddef>
#include <span>
#include <utility>

namespace rund::task {

class Group final {
public:
  class JoinState;

  explicit Group(const std::span<Handle> slots) noexcept : slots_(slots) {}
  Group(const Group &) = delete;
  Group &operator=(const Group &) = delete;
  Group(Group &&) = delete;
  Group &operator=(Group &&) = delete;

  template <typename Work>
  [[nodiscard]] Handle spawn(const char *const name, Work &&work) {
    if (join_active_ || size_ == slots_.size()) {
      return Handle{};
    }
    Handle handle = task::spawn(name, std::forward<Work>(work));
    if (handle) {
      slots_[size_++] = handle;
    }
    return handle;
  }

  template <typename Work> [[nodiscard]] Handle spawn(Work &&work) {
    return spawn("task", std::forward<Work>(work));
  }

  [[nodiscard]] JoinState begin_join() noexcept;

  [[nodiscard]] Task<Status> join();

  [[nodiscard]] std::span<const Handle> handles() const noexcept {
    return slots_.first(size_);
  }

  void clear() noexcept {
    if (!join_active_) {
      size_ = 0u;
    }
  }

  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0u; }
  [[nodiscard]] std::size_t capacity() const noexcept { return slots_.size(); }

private:
  std::span<Handle> slots_{};
  std::size_t size_{};
  bool join_active_ = false;
};

class Group::JoinState final {
public:
  JoinState(const JoinState &) = delete;
  JoinState &operator=(const JoinState &) = delete;

  JoinState(JoinState &&other) noexcept
      : group_(std::exchange(other.group_, nullptr)), index_(other.index_),
        code_(other.code_) {
    other.index_ = 0u;
    other.code_ = ReasonCode::TaskInvalid;
  }

  JoinState &operator=(JoinState &&) = delete;

  [[nodiscard]] bool pending() const noexcept {
    return group_ != nullptr && index_ < group_->size_;
  }

  [[nodiscard]] Handle current() const noexcept {
    return pending() ? group_->slots_[index_] : Handle{};
  }

  void advance(const Status joined) noexcept {
    if (!pending()) {
      return;
    }
    if (!joined && code_ == ReasonCode::Ok) {
      code_ = joined.code();
    }
    ++index_;
  }

  [[nodiscard]] Status finish() noexcept {
    if (group_ == nullptr) {
      return Status::fail(ReasonCode::TaskInvalid);
    }
    if (pending()) {
      return Status::fail(ReasonCode::TaskInvalid);
    }
    group_->size_ = 0u;
    group_->join_active_ = false;
    group_ = nullptr;
    return code_ == ReasonCode::Ok ? Status::success() : Status::fail(code_);
  }

private:
  friend class Group;

  explicit JoinState(Group *const group) noexcept
      : group_(group) {}

  Group *group_ = nullptr;
  std::size_t index_ = 0u;
  ReasonCode code_ = ReasonCode::Ok;
};

inline Group::JoinState Group::begin_join() noexcept {
  if (join_active_) {
    return JoinState{nullptr};
  }
  join_active_ = true;
  return JoinState{this};
}

inline Task<Status> Group::join() {
  JoinState joining = begin_join();
  while (joining.pending()) {
    joining.advance(co_await joining.current());
  }
  co_return joining.finish();
}

} // namespace rund::task
