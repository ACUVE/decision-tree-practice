add_executable(testout test_main.cpp)
target_link_libraries(testout ${CMAKE_DL_LIBS})
target_include_directories(testout PRIVATE ./include)
target_link_libraries(testout testso)
