cmake_minimum_required(VERSION 3.18)

project(test_debayer)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Modify if yours is different.  This one is Ubuntu:
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/lib/x86_64-linux-gnu/cmake/opencv4/)

# Find OpenCV package
find_package(OpenCV 4.2 REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

find_package(CUDAToolkit REQUIRED)
# #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_75) # Adjust architecture

find_package(libdebayer REQUIRED)
find_package(libdebayercpp REQUIRED)

# App
add_executable(test_debayer src/test_debayer.cpp)
target_link_libraries(test_debayer
    ${OpenCV_LIBS}
    libdebayercpp::debayercpp
    CUDA::cudart
)
target_include_directories(test_debayer PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

install(TARGETS test_debayer
        RUNTIME DESTINATION bin)

include(CTest)
if(BUILD_TESTING)
    add_executable(metrics_test tests/metrics_test.cpp)
    target_include_directories(metrics_test PRIVATE src ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(metrics_test PRIVATE ${OpenCV_LIBS})
    add_test(NAME metrics_test COMMAND metrics_test)
endif()
