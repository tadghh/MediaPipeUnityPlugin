# Copyright 2025 Custom Implementation
# BUILD file for PoseLandmarkerWithYolo graph

load("//mediapipe/framework:mediapipe_register_type.bzl", "mediapipe_register_type")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "pose_landmarker_with_yolo_graph",
    srcs = ["pose_landmarker_with_yolo_graph.cc"],
    deps = [
        "//mediapipe/tasks/cc/vision/pose_landmarker_with_yolo/proto:pose_landmarker_with_yolo_graph_options_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/tasks/cc/core:model_asset_bundle_resources",
        "//mediapipe/tasks/cc/core:model_task_graph",
        "//mediapipe/tasks/cc/vision/pose_landmarker:pose_landmarker_graph",
        "//mediapipe/tasks/cc/vision/pose_landmarker/proto:pose_landmarker_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/object_detector:object_detector_graph",
        "//mediapipe/tasks/cc/vision/object_detector/proto:object_detector_options_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

# Library that includes all dependencies for using the graph
cc_library(
    name = "pose_landmarker_with_yolo",
    deps = [
        ":pose_landmarker_with_yolo_graph",
        "//mediapipe/tasks/cc/vision/pose_landmarker:pose_landmarker_graph",
        "//mediapipe/tasks/cc/vision/object_detector:object_detector_graph",
    ],
)
