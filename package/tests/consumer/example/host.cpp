#include <rund/host.hpp>
#include <rund/session.hpp>

#include <string>
#include <utility>

int main() {
  int operation = 0;
  std::string path{};
  const rund::Session::Result hosted =
      rund::run(rund::SessionConfig{.workers = 1u}, [&] {
        auto value = rund::host::env::get("PATH");
        if (!value) {
          operation = value.exit_code();
          return;
        }
        path = std::move(*value);
      });
  if (!hosted) {
    return hosted.exit_code();
  }
  if (operation != 0) {
    return operation;
  }
  (void)path;
  return 0;
}
