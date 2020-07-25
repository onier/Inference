set(DEPENDS gl glu glew cuda gstreamer-1.0 gstreamer-pbutils-1.0 gstreamer-app-1.0)
include_directories(
        include/camera
        include/codec
        include/cuda
        include/display
        include/experimental
        include/image
        include/input
        include/network
        include/python
        include/threads
        include/video
)
include(cuda_library)