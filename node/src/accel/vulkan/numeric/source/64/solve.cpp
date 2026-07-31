#include "../../source.hpp"

#include "solve/direct.hpp"
#include "solve/factor.hpp"

#include <string>

namespace rund::node::accel::detail {

[[nodiscard]] std::string SolveSource64() {
  std::string out = NumericBaseSource64();
  out.reserve(out.size() + source::lane64::solve::Factor.size() +
              source::lane64::solve::Direct.size());
  out.append(source::lane64::solve::Factor);
  out.append(source::lane64::solve::Direct);
  return out;
}

} // namespace rund::node::accel::detail
