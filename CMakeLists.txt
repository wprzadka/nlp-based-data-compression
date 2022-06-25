cmake_minimum_required(VERSION 3.21)
project(ANS)

set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME}
        nlohmann/json.hpp
        src/rans.cpp
        src/tokenizer.cpp
        src/main.cpp)

add_subdirectory(googletest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

add_executable(${PROJECT_NAME}_tests
        src/rans.cpp
        tests/rans_tests.cpp
)
target_link_libraries(${PROJECT_NAME}_tests gtest gtest_main)