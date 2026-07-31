#pragma once

#include "../../../host/io/validation.hpp"

namespace rund::task {
class IoOp;
}

namespace rund::node::scheduler_io {

[[nodiscard]] task::IoOp WaitAdmittedFd(::rund::host::io::detail::FdIdentity fd,
                                        short interest) noexcept;

} // namespace rund::node::scheduler_io
