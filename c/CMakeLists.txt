cmake_minimum_required(VERSION 3.18)
project(libdebayer VERSION 0.1 LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

add_library(debayer SHARED src/debayer.cu src/debayer_kernel.cu)
target_include_directories(debayer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/libdebayer>)
target_link_libraries(debayer PUBLIC CUDA::cudart)
set_target_properties(debayer PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES
    CUDA_RUNTIME_LIBRARY Shared
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER include/debayer.h)

install(TARGETS debayer EXPORT libdebayerTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libdebayer)
install(EXPORT libdebayerTargets FILE libdebayerTargets.cmake
    NAMESPACE libdebayer:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libdebayer)
configure_package_config_file(cmake/libdebayerConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/libdebayerConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libdebayer)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/libdebayerConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libdebayerConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/libdebayerConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libdebayer)
# Keep explicit libdebayer_DIR=<prefix>/lib/cmake consumers working.
install(FILES cmake/libdebayerLegacyConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake RENAME libdebayerConfig.cmake)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libdebayerConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake)
configure_file(libdebayer.pc.in libdebayer.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libdebayer.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

if(BUILD_TESTING)
    add_executable(debayer_regression tests/regression.cu)
    target_link_libraries(debayer_regression PRIVATE debayer)
    set_target_properties(debayer_regression PROPERTIES
        CXX_STANDARD 17 CUDA_STANDARD 17 CUDA_RUNTIME_LIBRARY Shared)
    add_test(NAME debayer_cuda_regression COMMAND debayer_regression)
    add_executable(debayer_c_api_header tests/c_api_header.c)
    target_link_libraries(debayer_c_api_header PRIVATE debayer)
    add_test(NAME debayer_c_api_header COMMAND debayer_c_api_header)
endif()
