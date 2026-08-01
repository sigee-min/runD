#include <rund/session.hpp>
#include <rund/task.hpp>

int main() {
  int value = 0;
  rund::task::Status task_result = rund::task::Status::success();
  const rund::Session::Result hosted =
      rund::run(rund::SessionConfig{.workers = 1u}, [&] {
        const rund::task::Handle task =
            rund::task::spawn("set-value", [&] { value = 42; });
        if (!task) {
          task_result = rund::task::Status::fail(task.code());
          return;
        }
        task_result = rund::task::join(task);
      });
  if (!hosted) {
    return hosted.exit_code();
  }
  if (!task_result) {
    return task_result.exit_code();
  }
  return value == 42 ? 0 : 2;
}
