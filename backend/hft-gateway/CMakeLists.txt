cmake_minimum_required(VERSION 3.15)
project(hft_gateway LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------------------------
# Includes
# ------------------------------------------------------------------------------
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ------------------------------------------------------------------------------
# Core sources (library)
# ------------------------------------------------------------------------------
set(HFTGW_SOURCES
    src/md_snapshot.cpp
    src/order_router.cpp
    src/risk_wall.cpp
)

add_library(hftgw STATIC ${HFTGW_SOURCES})
target_include_directories(hftgw PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ------------------------------------------------------------------------------
# Main gateway binary
# ------------------------------------------------------------------------------
add_executable(hft_gateway
    src/main_gateway.cpp
)
target_link_libraries(hft_gateway PRIVATE hftgw)

# ------------------------------------------------------------------------------
# Demo binaries for shared memory rings
# ------------------------------------------------------------------------------
add_executable(shm_md_ringbuf_demo src/shm_md_ringbuf_demo.cpp)
target_link_libraries(shm_md_ringbuf_demo PRIVATE hftgw)

add_executable(shm_orders_in_demo src/shm_orders_in_demo.cpp)
target_link_libraries(shm_orders_in_demo PRIVATE hftgw)

add_executable(shm_fills_out_demo src/shm_fills_out_demo.cpp)
target_link_libraries(shm_fills_out_demo PRIVATE hftgw)

add_executable(shm_heartbeat_demo src/shm_heartbeat_demo.cpp)
target_link_libraries(shm_heartbeat_demo PRIVATE hftgw)

add_executable(shm_risk_wall_demo src/shm_risk_wall_demo.cpp)
target_link_libraries(shm_risk_wall_demo PRIVATE hftgw)

# ------------------------------------------------------------------------------
# Unit tests (simple assert.hpp style)
# ------------------------------------------------------------------------------
enable_testing()

add_executable(test_md_snapshot tests/test_md_snapshot.cpp)
target_link_libraries(test_md_snapshot PRIVATE hftgw)
add_test(NAME md_snapshot COMMAND test_md_snapshot)

add_executable(test_order_router tests/test_order_router.cpp)
target_link_libraries(test_order_router PRIVATE hftgw)
add_test(NAME order_router COMMAND test_order_router)

add_executable(test_risk_wall tests/test_risk_wall.cpp)
target_link_libraries(test_risk_wall PRIVATE hftgw)
add_test(NAME risk_wall COMMAND test_risk_wall)

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------
# Default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()