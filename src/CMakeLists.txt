add_library(testso SHARED decision.cpp)
set_target_properties(testso PROPERTIES CXX_VISIBILITY_PRESET internal)
target_include_directories(testso SYSTEM PRIVATE ../include)
target_include_directories(testso PUBLIC ./include)
