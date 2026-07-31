#include <rund/rund.hpp>

#include <type_traits>

static_assert(std::is_class_v<rund::Session>);
static_assert(std::is_class_v<rund::StableHash>);
static_assert(std::is_enum_v<rund::compute::Backend>);
static_assert(std::is_class_v<rund::net::Address>);
static_assert(std::is_class_v<rund::evidence::Numeric>);
static_assert(std::is_class_v<rund::task::Status>);
static_assert(std::is_class_v<rund::host::io::OpenOptions>);
static_assert(std::is_class_v<rund::storage::Budget>);
