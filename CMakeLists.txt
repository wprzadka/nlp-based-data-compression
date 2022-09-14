cmake_minimum_required(VERSION 3.0)
project(nlp_based_compression)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_PREFIX_PATH "venv/lib/python3.8/site-packages/torch/share/cmake")
#set(CMAKE_PREFIX_PATH "/libtorch/")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${PROJECT_NAME}
        nlohmann/json.hpp
        src/rans.cpp
        src/tokenizer.cpp
        src/predictor.cpp
        src/main.cpp)

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

#cmake -Wno-dev -DCMAKE_PREFIX_PATH=/libtorch/ ..
#cmake --build . --config Release

# TESTS
add_subdirectory(googletest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

add_executable(${PROJECT_NAME}_tests
        src/rans.cpp
        tests/rans_tests.cpp
)
target_link_libraries(${PROJECT_NAME}_tests gtest gtest_main)
