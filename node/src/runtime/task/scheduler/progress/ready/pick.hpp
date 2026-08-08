#pragma once

#include <cstdint>

namespace rund::node {

enum class ReadyPickDisposition : std::uint8_t {
  None,
  Task,
  Blocked,
  Activity,
};

class ReadyPick final {
public:
  [[nodiscard]] static constexpr ReadyPick none() noexcept {
    return ReadyPick{ReadyPickDisposition::None, 0u};
  }

  [[nodiscard]] static constexpr ReadyPick
  task(const std::uint64_t task_id) noexcept {
    return task_id == 0u ? none()
                         : ReadyPick{ReadyPickDisposition::Task, task_id};
  }

  [[nodiscard]] static constexpr ReadyPick blocked() noexcept {
    return ReadyPick{ReadyPickDisposition::Blocked, 0u};
  }

  [[nodiscard]] static constexpr ReadyPick activity() noexcept {
    return ReadyPick{ReadyPickDisposition::Activity, 0u};
  }

  [[nodiscard]] constexpr ReadyPickDisposition disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr std::uint64_t task_id() const noexcept {
    return task_id_;
  }

private:
  constexpr ReadyPick(const ReadyPickDisposition disposition,
                      const std::uint64_t task_id) noexcept
      : disposition_(disposition), task_id_(task_id) {}

  ReadyPickDisposition disposition_ = ReadyPickDisposition::None;
  std::uint64_t task_id_ = 0u;
};

} // namespace rund::node
