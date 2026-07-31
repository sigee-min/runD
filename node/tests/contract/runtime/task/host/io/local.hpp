#pragma once

#include <string>

namespace runtime_task_host_io {

struct Fd {
  int native = -1;
  bool owned = true;

  Fd() = default;
  explicit Fd(int value) noexcept : native(value) {}
  Fd(const Fd &) = delete;
  Fd &operator=(const Fd &) = delete;
  ~Fd();

  void release() noexcept { owned = false; }
};

struct Temp {
  std::string path{};

  ~Temp();
};

[[nodiscard]] std::string Path(const char *suffix);

void Surface();
void Admission();
void Replay();
void Order();
void Signal();

} // namespace runtime_task_host_io
