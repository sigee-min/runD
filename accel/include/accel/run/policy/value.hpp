#pragma once

namespace rund::node::accel {

struct RunPolicy final {
  bool staged = true;
};

struct RunChoice final {
  bool use_accel = false;
  const char *reason = "accel_run_choice_invalid";
};

} // namespace rund::node::accel
