#include "../source.hpp"

#include "solve/direct.hpp"
#include "solve/factor.hpp"

#include <string>

namespace rund::node::accel::detail {

[[nodiscard]] std::string SolveSource() {
  std::string out = NumericBaseSource();
  out.reserve(out.size() + source::solve::Factor.size() +
              source::solve::Direct.size());
  out.append(source::solve::Factor);
  out.append(source::solve::Direct);
  return out;
}

} // namespace rund::node::accel::detail
