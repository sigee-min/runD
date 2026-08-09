#pragma once

[[nodiscard]] inline std::string RangeFunctionName(const RangeExec &execution) {
  const char *const operation =
      execution.operation() == RangeOp::Minimum   ? "min"
      : execution.operation() == RangeOp::Maximum ? "max"
                                                  : "sum";
  std::string name = "rund_range_";
  name += operation;
  name += execution.signed_extrema() ? "_i" : "_u";
  name += execution.wide_elements() ? "64" : "32";
  return name;
}
