#include "local.hpp"

int RunRuntimeTaskHostIoContract() {
  runtime_task_host_io::Surface();
  runtime_task_host_io::Admission();
  runtime_task_host_io::Replay();
  runtime_task_host_io::Order();
  runtime_task_host_io::Signal();
  return 0;
}
