cmake_minimum_required(VERSION 3.18)
project(libdebayer-cpp VERSION 0.1 LANGUAGES CXX CUDA)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)
find_package(libdebayer CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(debayercpp SHARED src/debayer_cpp.cu)
target_include_directories(debayercpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(debayercpp PUBLIC libdebayer::debayer CUDA::cudart)
set_target_properties(debayercpp PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES
    CUDA_RUNTIME_LIBRARY Shared
    VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER include/debayer_cpp.h)

install(TARGETS debayercpp EXPORT libdebayercppTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libdebayercpp)
install(EXPORT libdebayercppTargets FILE libdebayercppTargets.cmake
    NAMESPACE libdebayercpp:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libdebayercpp)
configure_package_config_file(libdebayercppConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/libdebayercppConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libdebayercpp)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/libdebayercppConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libdebayercppConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/libdebayercppConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libdebayercpp)
install(FILES libdebayercppCompatConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake RENAME libdebayercppConfig.cmake)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libdebayercppConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake)
configure_file(libdebayercpp.pc.in libdebayercpp.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libdebayercpp.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

if(BUILD_TESTING)
    add_executable(wrapper_test tests/wrapper_test.cpp)
    target_link_libraries(wrapper_test PRIVATE debayercpp)
    add_test(NAME wrapper_test COMMAND wrapper_test)
endif()
