// ros
#include "pose_estimation.hpp"
#include <apriltag_msgs/msg/april_tag_detection.hpp>
#include <apriltag_msgs/msg/april_tag_detection_array.hpp>
#ifdef cv_bridge_HPP
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_ros/transform_broadcaster.h>

// Add these includes for approximate time synchronization
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

// apriltag
#include "tag_functions.hpp"
#include <apriltag.h>


#define IF(N, V) \
    if(assign_check(parameter, N, V)) continue;

template<typename T>
void assign(const rclcpp::Parameter& parameter, T& var)
{
    var = parameter.get_value<T>();
}

template<typename T>
void assign(const rclcpp::Parameter& parameter, std::atomic<T>& var)
{
    var = parameter.get_value<T>();
}

template<typename T>
bool assign_check(const rclcpp::Parameter& parameter, const std::string& name, T& var)
{
    if(parameter.get_name() == name) {
        assign(parameter, var);
        return true;
    }
    return false;
}

rcl_interfaces::msg::ParameterDescriptor
descr(const std::string& description, const bool& read_only = false)
{
    rcl_interfaces::msg::ParameterDescriptor descr;

    descr.description = description;
    descr.read_only = read_only;

    return descr;
}

class AprilTagNode : public rclcpp::Node {
public:
    using ImageMsg = sensor_msgs::msg::Image;
    using CameraInfoMsg = sensor_msgs::msg::CameraInfo;
    using ApproximatePolicy = message_filters::sync_policies::ApproximateTime<ImageMsg, CameraInfoMsg>;
    using Synchronizer = message_filters::Synchronizer<ApproximatePolicy>;

    AprilTagNode(const rclcpp::NodeOptions& options);
    ~AprilTagNode() override;

private:
    const OnSetParametersCallbackHandle::SharedPtr cb_parameter;

    apriltag_family_t* tf;
    apriltag_detector_t* const td;

    // parameter
    std::mutex mutex;
    double tag_edge_size;
    std::atomic<int> max_hamming;
    std::atomic<bool> profile;
    std::unordered_map<int, std::string> tag_frames;
    std::unordered_map<int, double> tag_sizes;

    std::function<void(apriltag_family_t*)> tf_destructor;

    // Replace the image_transport subscriber with message_filters subscribers
    message_filters::Subscriber<ImageMsg> sub_image;
    message_filters::Subscriber<CameraInfoMsg> sub_info;
    std::shared_ptr<Synchronizer> sync;

    const rclcpp::Publisher<apriltag_msgs::msg::AprilTagDetectionArray>::SharedPtr pub_detections;
    tf2_ros::TransformBroadcaster tf_broadcaster;

    pose_estimation_f estimate_pose = nullptr;

    // Debug counters
    size_t total_frames_processed = 0;
    size_t total_tags_detected = 0;

    void onCamera(const ImageMsg::ConstSharedPtr& msg_img, const CameraInfoMsg::ConstSharedPtr& msg_ci);

    rcl_interfaces::msg::SetParametersResult onParameter(const std::vector<rclcpp::Parameter>& parameters);
};

RCLCPP_COMPONENTS_REGISTER_NODE(AprilTagNode)


AprilTagNode::AprilTagNode(const rclcpp::NodeOptions& options)
  : Node("apriltag", options),
    // parameter
    cb_parameter(add_on_set_parameters_callback(std::bind(&AprilTagNode::onParameter, this, std::placeholders::_1))),
    td(apriltag_detector_create()),
    // Initialize message_filters subscribers instead of image_transport
    sub_image(this, this->get_node_topics_interface()->resolve_topic_name("image_rect"), 
              rmw_qos_profile_sensor_data),
    sub_info(this, this->get_node_topics_interface()->resolve_topic_name("camera_info"), 
             rmw_qos_profile_sensor_data),
    pub_detections(create_publisher<apriltag_msgs::msg::AprilTagDetectionArray>("detections", rclcpp::QoS(1))),
    tf_broadcaster(this)
{
    // Initialize synchronizer with queue size and optional maximum time difference
    // The queue size determines how many messages are stored for synchronization
    // You can adjust these parameters based on your needs
    const int queue_size = declare_parameter("sync_queue_size", 10, 
                                           descr("Queue size for approximate time synchronizer"));
    const double max_interval = declare_parameter("sync_max_interval", 0.1, 
                                                descr("Maximum time difference between messages for synchronization (seconds)"));
    
    // Create synchronizer with approximate time policy
    sync = std::make_shared<Synchronizer>(ApproximatePolicy(queue_size), sub_image, sub_info);
    
    // Optionally set maximum interval age for synchronization
    if (max_interval > 0.0) {
        sync->setMaxIntervalDuration(rclcpp::Duration::from_seconds(max_interval));
    }
    
    // Add individual callbacks for debugging
    sub_image.registerCallback([this](const ImageMsg::ConstSharedPtr& msg) {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Received image message (size: %dx%d, encoding: %s)", 
                             msg->width, msg->height, msg->encoding.c_str());
    });
    
    sub_info.registerCallback([this](const CameraInfoMsg::ConstSharedPtr& msg) {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Received camera info message (size: %dx%d)", 
                             msg->width, msg->height);
    });
    
    // Register callback
    sync->registerCallback(std::bind(&AprilTagNode::onCamera, this, 
                                   std::placeholders::_1, std::placeholders::_2));

    // read-only parameters
    const std::string tag_family = declare_parameter("family", "36h11", descr("tag family", true));
    tag_edge_size = declare_parameter("size", 1.0, descr("default tag size", true));

    // get tag names, IDs and sizes
    const auto ids = declare_parameter("tag.ids", std::vector<int64_t>{}, descr("tag ids", true));
    const auto frames = declare_parameter("tag.frames", std::vector<std::string>{}, descr("tag frame names per id", true));
    const auto sizes = declare_parameter("tag.sizes", std::vector<double>{}, descr("tag sizes per id", true));

    // get method for estimating tag pose
    const std::string& pose_estimation_method =
        declare_parameter("pose_estimation_method", "pnp",
                          descr("pose estimation method: \"pnp\" (more accurate) or \"homography\" (faster), "
                                "set to \"\" (empty) to disable pose estimation",
                                true));

    if(!pose_estimation_method.empty()) {
        if(pose_estimation_methods.count(pose_estimation_method)) {
            estimate_pose = pose_estimation_methods.at(pose_estimation_method);
            RCLCPP_INFO(get_logger(), "Using pose estimation method: %s", pose_estimation_method.c_str());
        }
        else {
            RCLCPP_ERROR_STREAM(get_logger(), "Unknown pose estimation method '" << pose_estimation_method << "'.");
        }
    } else {
        RCLCPP_INFO(get_logger(), "Pose estimation disabled");
    }

    // detector parameters in "detector" namespace
    declare_parameter("detector.threads", td->nthreads, descr("number of threads"));
    declare_parameter("detector.decimate", td->quad_decimate, descr("decimate resolution for quad detection"));
    declare_parameter("detector.blur", td->quad_sigma, descr("sigma of Gaussian blur for quad detection"));
    declare_parameter("detector.refine", td->refine_edges, descr("snap to strong gradients"));
    declare_parameter("detector.sharpening", td->decode_sharpening, descr("sharpening of decoded images"));
    declare_parameter("detector.debug", td->debug, descr("write additional debugging images to working directory"));

    declare_parameter("max_hamming", 0, descr("reject detections with more corrected bits than allowed"));
    declare_parameter("profile", false, descr("print profiling information to stdout"));

    if(!frames.empty()) {
        if(ids.size() != frames.size()) {
            throw std::runtime_error("Number of tag ids (" + std::to_string(ids.size()) + ") and frames (" + std::to_string(frames.size()) + ") mismatch!");
        }
        for(size_t i = 0; i < ids.size(); i++) { 
            tag_frames[ids[i]] = frames[i]; 
            RCLCPP_INFO(get_logger(), "Tag ID %ld mapped to frame %s", ids[i], frames[i].c_str());
        }
    }

    if(!sizes.empty()) {
        // use tag specific size
        if(ids.size() != sizes.size()) {
            throw std::runtime_error("Number of tag ids (" + std::to_string(ids.size()) + ") and sizes (" + std::to_string(sizes.size()) + ") mismatch!");
        }
        for(size_t i = 0; i < ids.size(); i++) { 
            tag_sizes[ids[i]] = sizes[i]; 
            RCLCPP_INFO(get_logger(), "Tag ID %ld has size %f", ids[i], sizes[i]);
        }
    }

    if(tag_fun.count(tag_family)) {
        tf = tag_fun.at(tag_family).first();
        tf_destructor = tag_fun.at(tag_family).second;
        apriltag_detector_add_family(td, tf);
        RCLCPP_INFO(get_logger(), "Added tag family: %s", tag_family.c_str());
    }
    else {
        throw std::runtime_error("Unsupported tag family: " + tag_family);
    }

    // Log detector parameters
    RCLCPP_INFO(get_logger(), "Detector parameters:");
    RCLCPP_INFO(get_logger(), "  threads: %d", td->nthreads);
    RCLCPP_INFO(get_logger(), "  decimate: %f", td->quad_decimate);
    RCLCPP_INFO(get_logger(), "  blur: %f", td->quad_sigma);
    RCLCPP_INFO(get_logger(), "  refine: %d", td->refine_edges);
    RCLCPP_INFO(get_logger(), "  debug: %d", td->debug);
    RCLCPP_INFO(get_logger(), "  max_hamming: %d", max_hamming.load());
    RCLCPP_INFO(get_logger(), "  tag_edge_size: %f", tag_edge_size);

    // Log the topics we're subscribing to
    RCLCPP_INFO(get_logger(), "Subscribing to image topic: %s", 
                this->get_node_topics_interface()->resolve_topic_name("image_rect").c_str());
    RCLCPP_INFO(get_logger(), "Subscribing to camera_info topic: %s", 
                this->get_node_topics_interface()->resolve_topic_name("camera_info").c_str());
    
    RCLCPP_INFO(get_logger(), "AprilTag detector initialized with approximate time synchronization");
}

AprilTagNode::~AprilTagNode()
{
    RCLCPP_INFO(get_logger(), "Shutting down AprilTag detector. Processed %zu frames, detected %zu total tags", 
                total_frames_processed, total_tags_detected);
    apriltag_detector_destroy(td);
    tf_destructor(tf);
}

void AprilTagNode::onCamera(const ImageMsg::ConstSharedPtr& msg_img,
                            const CameraInfoMsg::ConstSharedPtr& msg_ci)
{
    RCLCPP_INFO(get_logger(), "Received synchronized image and camera info");
    total_frames_processed++;
    
    // Debug image properties
    RCLCPP_DEBUG(get_logger(), "Image size: %dx%d, encoding: %s", 
                 msg_img->width, msg_img->height, msg_img->encoding.c_str());
    
    // camera intrinsics for rectified images
    const std::array<double, 4> intrinsics = {msg_ci->p[0], msg_ci->p[5], msg_ci->p[2], msg_ci->p[6]};
    
    RCLCPP_DEBUG(get_logger(), "Camera intrinsics: fx=%f, fy=%f, cx=%f, cy=%f", 
                 intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]);

    // check for valid intrinsics
    const bool calibrated = msg_ci->width && msg_ci->height &&
                            intrinsics[0] && intrinsics[1] && intrinsics[2] && intrinsics[3];

    if(!calibrated) {
        RCLCPP_WARN(get_logger(), "Camera is not calibrated! width=%d, height=%d, fx=%f, fy=%f, cx=%f, cy=%f",
                    msg_ci->width, msg_ci->height, intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]);
    }

    if(estimate_pose != nullptr && !calibrated) {
        RCLCPP_WARN_STREAM(get_logger(), "The camera is not calibrated! Set 'pose_estimation_method' to \"\" (empty) to disable pose estimation and this warning.");
    }

    // convert to 8bit monochrome image
    cv::Mat img_uint8;
    try {
        img_uint8 = cv_bridge::toCvShare(msg_img, "mono8")->image;
        RCLCPP_DEBUG(get_logger(), "Successfully converted image to mono8");
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    image_u8_t im{img_uint8.cols, img_uint8.rows, img_uint8.cols, img_uint8.data};

    // detect tags
    mutex.lock();
    zarray_t* detections = apriltag_detector_detect(td, &im);
    mutex.unlock();

    int n_detections = zarray_size(detections);
    RCLCPP_INFO(get_logger(), "Detected %d tags in frame %zu", n_detections, total_frames_processed);
    total_tags_detected += n_detections;

    if(profile)
        timeprofile_display(td->tp);

    apriltag_msgs::msg::AprilTagDetectionArray msg_detections;
    msg_detections.header = msg_img->header;

    std::vector<geometry_msgs::msg::TransformStamped> tfs;

    for(int i = 0; i < n_detections; i++) {
        apriltag_detection_t* det;
        zarray_get(detections, i, &det);

        RCLCPP_INFO(get_logger(), "Detection %d: id=%d (family=%s), hamming=%d, margin=%.3f, center=(%.1f,%.1f)",
                    i, det->id, det->family->name,
                    det->hamming, det->decision_margin,
                    det->c[0], det->c[1]);

        // ignore untracked tags
        if(!tag_frames.empty() && !tag_frames.count(det->id)) { 
            RCLCPP_WARN(get_logger(), "Ignoring untracked tag id %d", det->id);
            continue; 
        }

        // reject detections with more corrected bits than allowed
        if(det->hamming > max_hamming) { 
            RCLCPP_WARN(get_logger(), "Rejecting tag id %d with hamming=%d (max_hamming=%d)", 
                        det->id, det->hamming, max_hamming.load());
            continue; 
        }

        // detection
        apriltag_msgs::msg::AprilTagDetection msg_detection;
        msg_detection.family = std::string(det->family->name);
        msg_detection.id = det->id;
        msg_detection.hamming = det->hamming;
        msg_detection.decision_margin = det->decision_margin;
        msg_detection.centre.x = det->c[0];
        msg_detection.centre.y = det->c[1];
        std::memcpy(msg_detection.corners.data(), det->p, sizeof(double) * 8);
        std::memcpy(msg_detection.homography.data(), det->H->data, sizeof(double) * 9);
        msg_detections.detections.push_back(msg_detection);

        // 3D orientation and position
        if(estimate_pose != nullptr && calibrated) {
            geometry_msgs::msg::TransformStamped tf;
            tf.header = msg_img->header;
            // set child frame name by generic tag name or configured tag name
            tf.child_frame_id = tag_frames.count(det->id) ? tag_frames.at(det->id) : std::string(det->family->name) + ":" + std::to_string(det->id);
            const double size = tag_sizes.count(det->id) ? tag_sizes.at(det->id) : tag_edge_size;
            
            RCLCPP_DEBUG(get_logger(), "Estimating pose for tag %d with size %f", det->id, size);
            
            tf.transform = estimate_pose(det, intrinsics, size);
            
            RCLCPP_INFO(get_logger(), "Tag %d pose: position=(%.3f, %.3f, %.3f), quaternion=(%.3f, %.3f, %.3f, %.3f)",
                        det->id, 
                        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z,
                        tf.transform.rotation.x, tf.transform.rotation.y, 
                        tf.transform.rotation.z, tf.transform.rotation.w);
            
            tfs.push_back(tf);
        }
    }

    if(!msg_detections.detections.empty()) {
        RCLCPP_INFO(get_logger(), "Publishing %zu detections", msg_detections.detections.size());
        pub_detections->publish(msg_detections);
    } else {
        RCLCPP_DEBUG(get_logger(), "No detections to publish");
    }

    if(estimate_pose != nullptr && !tfs.empty()) {
        RCLCPP_INFO(get_logger(), "Broadcasting %zu transforms", tfs.size());
        tf_broadcaster.sendTransform(tfs);
    }

    apriltag_detections_destroy(detections);
    
    // Periodic summary
    if(total_frames_processed % 100 == 0) {
        RCLCPP_INFO(get_logger(), "Summary: Processed %zu frames, detected %zu total tags (%.2f tags/frame average)",
                    total_frames_processed, total_tags_detected, 
                    static_cast<double>(total_tags_detected) / static_cast<double>(total_frames_processed));
    }
}

rcl_interfaces::msg::SetParametersResult
AprilTagNode::onParameter(const std::vector<rclcpp::Parameter>& parameters)
{
    rcl_interfaces::msg::SetParametersResult result;

    mutex.lock();

    for(const rclcpp::Parameter& parameter : parameters) {
        RCLCPP_INFO(get_logger(), "Setting parameter: %s", parameter.get_name().c_str());

        IF("detector.threads", td->nthreads)
        IF("detector.decimate", td->quad_decimate)
        IF("detector.blur", td->quad_sigma)
        IF("detector.refine", td->refine_edges)
        IF("detector.sharpening", td->decode_sharpening)
        IF("detector.debug", td->debug)
        IF("max_hamming", max_hamming)
        IF("profile", profile)
    }

    mutex.unlock();

    result.successful = true;

    return result;
}