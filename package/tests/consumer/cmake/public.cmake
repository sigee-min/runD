add_library(rund_package_public_impl STATIC
            ../fixtures/public/library.cpp)
target_include_directories(rund_package_public_impl
                           PUBLIC ../fixtures/public)
target_link_libraries(rund_package_public_impl PUBLIC runD::sdk)

add_executable(rund_package_public_consumer
               ../fixtures/public/app.cpp)
target_link_libraries(rund_package_public_consumer
                      PRIVATE rund_package_public_impl)
rund_register_consumer(rund_package_public_consumer 0 general)
