# Copyright 2025 Custom Implementation

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

mediapipe_proto_library(
    name = "pose_landmarker_with_yolo_graph_options_proto",
    srcs = ["pose_landmarker_with_yolo_graph_options.proto"],
    deps = [
        "//mediapipe/tasks/cc/core/proto:base_options_proto",
        "//mediapipe/tasks/cc/vision/pose_landmarker/proto:pose_landmarker_graph_options_proto",
        "//mediapipe/tasks/cc/vision/object_detector/proto:object_detector_options_proto",
    ],
)
