set(DEPENDS rttr rapidjson boost xerces-c gl glu glew cuda gstreamer-1.0 gstreamer-pbutils-1.0 gstreamer-app-1.0 glog)
include_directories(include/video
                    include/cuda
                    include/camera
                    include/display
                    include/codec
        )
include(cuda_library)