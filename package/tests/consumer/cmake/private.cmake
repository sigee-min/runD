add_library(rund_package_private_impl STATIC
            ../fixtures/private/link.cpp)
target_link_libraries(rund_package_private_impl PRIVATE runD::sdk)

add_executable(rund_package_private_consumer
               ../fixtures/private/consumer.cpp)
target_link_libraries(rund_package_private_consumer
                      PRIVATE rund_package_private_impl)
rund_register_consumer(rund_package_private_consumer 0 general)
