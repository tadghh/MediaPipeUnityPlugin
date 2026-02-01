/* Copyright 2025 Custom Implementation
 *
 * Combined PoseLandmarker + YOLO Object Detection Graph
 * Runs both models in parallel on the same input frame.
 */

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/core/model_asset_bundle_resources.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker_with_yolo/proto/pose_landmarker_with_yolo_graph_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace pose_landmarker_with_yolo {

namespace {

using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::core::ModelAssetBundleResources;

constexpr char kImageTag[] = "IMAGE";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kPoseLandmarksTag[] = "POSE_LANDMARKS";
constexpr char kPoseWorldLandmarksTag[] = "POSE_WORLD_LANDMARKS";
constexpr char kPoseSegmentationMaskTag[] = "POSE_SEGMENTATION_MASK";
constexpr char kObjectDetectionsTag[] = "OBJECT_DETECTIONS";

// Model filenames inside the .task bundle
constexpr char kPoseDetectorModelName[] = "pose_detector.tflite";
constexpr char kPoseLandmarksModelName[] = "pose_landmarks_detector.tflite";
constexpr char kYoloModelName[] = "yolo.tflite";

}  // namespace

// Combined graph that runs PoseLandmarker and YOLO ObjectDetector in parallel.
//
// Inputs:
//   IMAGE - Image to process
//   NORM_RECT - Optional normalized rect for ROI
//
// Outputs:
//   POSE_LANDMARKS - Pose landmarks for each detected person
//   POSE_WORLD_LANDMARKS - 3D world landmarks for each detected person
//   POSE_SEGMENTATION_MASK - Segmentation mask (optional)
//   OBJECT_DETECTIONS - YOLO object detections
//
// Example usage:
// node {
//   calculator: "mediapipe.tasks.vision.pose_landmarker_with_yolo.PoseLandmarkerWithYoloGraph"
//   input_stream: "IMAGE:image"
//   input_stream: "NORM_RECT:norm_rect"
//   output_stream: "POSE_LANDMARKS:pose_landmarks"
//   output_stream: "POSE_WORLD_LANDMARKS:pose_world_landmarks"
//   output_stream: "OBJECT_DETECTIONS:object_detections"
//   options {
//     [mediapipe.tasks.vision.pose_landmarker_with_yolo.proto.PoseLandmarkerWithYoloGraphOptions.ext] {
//       base_options {
//         model_asset { file_name: "pose_landmarker_with_yolo.task" }
//       }
//       pose_landmarker_options {
//         num_poses: 1
//         min_pose_detection_confidence: 0.5
//         min_pose_presence_confidence: 0.5
//         min_tracking_confidence: 0.5
//       }
//       object_detector_options {
//         max_results: 10
//         score_threshold: 0.5
//       }
//     }
//   }
// }
class PoseLandmarkerWithYoloGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    Graph graph;

    // Get the model bundle resources
    MP_ASSIGN_OR_RETURN(
        const ModelAssetBundleResources* model_bundle,
        CreateModelAssetBundleResources<
            proto::PoseLandmarkerWithYoloGraphOptions>(sc));

    // Input streams
    auto image_in = graph.In(kImageTag).Cast<Image>();
    auto norm_rect_in = graph.In(kNormRectTag).Cast<NormalizedRect>();

    // =========================================================
    // BRANCH 1: Pose Landmarker (runs in parallel)
    // =========================================================
    auto& pose_landmarker = graph.AddNode(
        "mediapipe.tasks.vision.pose_landmarker.PoseLandmarkerGraph");
    
    // Configure pose landmarker from our options
    auto& pose_options = 
        pose_landmarker.GetOptions<pose_landmarker::proto::PoseLandmarkerGraphOptions>();
    
    // Copy pose landmarker options from our combined options
    const auto& options = 
        sc->Options<proto::PoseLandmarkerWithYoloGraphOptions>();
    if (options.has_pose_landmarker_options()) {
      pose_options = options.pose_landmarker_options();
    }
    
    // Set the model bundle for pose landmarker
    pose_options.mutable_base_options()
        ->mutable_model_asset()
        ->set_file_name(sc->Options<proto::PoseLandmarkerWithYoloGraphOptions>()
                            .base_options()
                            .model_asset()
                            .file_name());

    // Connect inputs
    image_in >> pose_landmarker.In(kImageTag);
    norm_rect_in >> pose_landmarker.In(kNormRectTag);

    // Get pose outputs
    auto pose_landmarks = 
        pose_landmarker.Out(kPoseLandmarksTag)
            .Cast<std::vector<NormalizedLandmarkList>>();
    auto pose_world_landmarks = 
        pose_landmarker.Out(kPoseWorldLandmarksTag)
            .Cast<std::vector<LandmarkList>>();

    // =========================================================
    // BRANCH 2: YOLO Object Detector (runs in parallel)
    // =========================================================
    auto& object_detector = graph.AddNode(
        "mediapipe.tasks.vision.object_detector.ObjectDetectorGraph");
    
    // Configure object detector
    auto& detector_options = 
        object_detector.GetOptions<object_detector::proto::ObjectDetectorOptions>();
    
    // Copy object detector options from our combined options
    if (options.has_object_detector_options()) {
      detector_options = options.object_detector_options();
    }
    
    // Set the YOLO model path (within the bundle)
    detector_options.mutable_base_options()
        ->mutable_model_asset()
        ->set_file_name(kYoloModelName);

    // Connect the SAME image input (enables parallel execution!)
    image_in >> object_detector.In(kImageTag);

    // Get detection outputs
    auto object_detections = 
        object_detector.Out("DETECTIONS").Cast<std::vector<Detection>>();

    // =========================================================
    // Connect outputs to graph
    // =========================================================
    pose_landmarks >> graph.Out(kPoseLandmarksTag);
    pose_world_landmarks >> graph.Out(kPoseWorldLandmarksTag);
    object_detections >> graph.Out(kObjectDetectionsTag);

    return graph.GetConfig();
  }
};
// Balls in play yo
REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::vision::pose_landmarker_with_yolo::PoseLandmarkerWithYoloGraph);

}  // namespace pose_landmarker_with_yolo
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
