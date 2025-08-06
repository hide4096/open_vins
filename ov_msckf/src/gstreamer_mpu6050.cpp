#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <endian.h>

#include <opencv2/opencv.hpp>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "core/VioManager.h"
#include "sim/Simulator.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/dataset_reader.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

#define ENABLE_IMU(x) write_sysfs(sysfs_buffer_base + "enable", x)

const std::string sysfs_base = "/sys/bus/iio/devices/iio:device0/";
const std::string sysfs_buffer_base = sysfs_base + "buffer0/";
const std::string dev_path = "/dev/iio:device0";

using namespace ov_type;
using namespace ov_core;
using namespace ov_msckf;

// sysfsへ値を書き込む
bool write_sysfs(const std::string& path, const std::string& value) {
    std::ofstream ofs(path);
    if (!ofs) {
        std::perror(("open: " + path).c_str());
        return false;
    }
    ofs << value;
    ofs.close();
    return true;
}

// sysfsから値を読む
std::string read_sysfs(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) return "";
    std::string val;
    std::getline(ifs, val);
    return val;
}

typedef struct{
    int16_t ax,ay,az;
    int16_t gx,gy,gz;
    int64_t timestamp;
} IMUData_t;

typedef struct{
    float anglvel_scale;
    float accel_scale;
    int sampling_freq;
    int buffer_length;
} IMUInfo_t;

/**
 * @brief IMUの初期化を行う関数
 * @param info 初期化に必要なIMUの設定情報
 * @return 成功した場合はtrue、失敗した場合はfalseを返す
 */
bool init_IMU(IMUInfo_t info) {
    // 処理の失敗状態を追跡するフラグ
    bool isFailed = false;

    // 各設定値をsysfs経由でデバイスに書き込む
    isFailed |= !write_sysfs(sysfs_base + "sampling_frequency", std::to_string(info.sampling_freq));
    isFailed |= !write_sysfs(sysfs_base + "in_accel_scale", std::to_string(info.accel_scale));
    isFailed |= !write_sysfs(sysfs_base + "in_anglvel_scale", std::to_string(info.anglvel_scale));

    // 読み込みバッファの有効化
    isFailed |= !write_sysfs(sysfs_buffer_base + "in_accel_x_en", "1");
    isFailed |= !write_sysfs(sysfs_buffer_base + "in_accel_y_en", "1");
    isFailed |= !write_sysfs(sysfs_buffer_base + "in_accel_z_en", "1");
    isFailed |= !write_sysfs(sysfs_buffer_base + "in_temp_en", "0");
    isFailed |= !write_sysfs(sysfs_buffer_base + "in_anglvel_x_en", "1");
    isFailed |= !write_sysfs(sysfs_buffer_base + "in_anglvel_y_en", "1");
    isFailed |= !write_sysfs(sysfs_buffer_base + "in_anglvel_z_en", "1");
    isFailed |= !write_sysfs(sysfs_buffer_base + "in_timestamp_en", "1");

    // バッファサイズの設定
    isFailed |= !write_sysfs(sysfs_buffer_base + "length", std::to_string(info.buffer_length));

    // 全ての書き込みが成功したかどうかの結果を返す
    return !isFailed;
}

double ns_to_sec(int64_t ns){
    return ns / (1000. * 1000. * 1000.);
}

int main(int argc, char **argv) {
    std::shared_ptr<VioManager> sys;

    if (argc < 1) {
        return 1;
    }

    auto parser = std::make_shared<ov_core::YamlParser>(argv[1]);

    VioManagerOptions params;
    params.print_and_load(parser);
    params.num_opencv_threads = 0;
    params.use_multi_threading_pubs = false;
    params.use_multi_threading_subs = false;

    sys = std::make_shared<VioManager>(params);

    if (!parser->successful()) {
        PRINT_ERROR(RED "unable to parse all parameters, please fix\n" RESET);
        std::exit(EXIT_FAILURE);
    }
    
    // IMUの初期化
    ENABLE_IMU("0");
    
    IMUData_t imu_buffer[20];

    IMUInfo_t setting = {
        .anglvel_scale = 0.001064724,
        .accel_scale = 0.004785,
        .sampling_freq = 200,
        .buffer_length = 20,
    };

    if(!init_IMU(setting)){
        std::perror("failed to setting IMU");
        return 1;
    }

    // カメラの初期化
    std::string gstPipeline =
        "libcamerasrc ! "
        "video/x-raw,width=640,height=480,framerate=30/1 ! "
        "videoconvert ! "
        "video/x-raw, format=GRAY8 ! "
        "appsink drop=true sync=false";

    cv::VideoCapture cap(gstPipeline, cv::CAP_GSTREAMER);

    if(!cap.isOpened()){
        std::perror("failed to open camera");
        return 1;
    }

 
    // IMU読み取り開始
    ENABLE_IMU("1");
    int fd = open(dev_path.c_str(), O_RDONLY | O_NONBLOCK);
    if (fd < 0) {
        ENABLE_IMU("0");
        std::perror("open /dev/iio:device0");
        return 1;
    }

    cv::Mat frame;
    int count = 0;
    while(count < 30 * 60 * 2){
        // フレーム取得
        cap >> frame;
        auto now = std::chrono::system_clock::now();
        int64_t frame_timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()
        ).count();
        
        // フレームが空なら終了
        if(frame.empty()){
            std::perror("empty frame recieved");
            break;
        }
        
        // IMU取得
        ssize_t rd = read(fd, imu_buffer, sizeof(imu_buffer));

        // IMU取得失敗してたら終了
        if (rd < 0) {
            // データないだけの時はスキップ
            if(errno != EAGAIN && errno != EWOULDBLOCK){
                std::perror("read");
                break;
            }
        }

        // IMUデータを供給
        int length = rd / sizeof(IMUData_t);
        for(int i=0;i<length;i++){
            IMUData_t data = imu_buffer[i];
            double accelx = (int16_t)be16toh(data.ax) * setting.accel_scale;
            double accely = (int16_t)be16toh(data.ay) * setting.accel_scale;
            double accelz = (int16_t)be16toh(data.az) * setting.accel_scale;
            double anglevelx = (int16_t)be16toh(data.gx) * setting.anglvel_scale;
            double anglevely = (int16_t)be16toh(data.gy) * setting.anglvel_scale;
            double anglevelz = (int16_t)be16toh(data.gz) * setting.anglvel_scale;

            ov_core::ImuData imu_message;
            imu_message.timestamp = ns_to_sec(data.timestamp);
            imu_message.wm << anglevelx, anglevely, anglevelz;
            imu_message.am << accelx, accely, accelz;

            sys->feed_measurement_imu(imu_message);
        }
        // 画像を供給
        ov_core::CameraData cam_message;
        cam_message.timestamp = ns_to_sec(frame_timestamp);
        cam_message.sensor_ids.push_back(0);
        cam_message.images.push_back(frame);
        cam_message.masks.push_back(cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_8UC1));

        sys->feed_measurement_camera(cam_message);

        // 表示
        if(sys->initialized()){
            std::shared_ptr<State> state = sys->get_state();
            Eigen::Vector3d position = state->_imu->pos();

            // UDPブロードキャスト送信
            static int udp_sock = -1;
            static sockaddr_in udp_addr;
            if (udp_sock == -1) {
                udp_sock = socket(AF_INET, SOCK_DGRAM, 0);
                int broadcastEnable = 1;
                setsockopt(udp_sock, SOL_SOCKET, SO_BROADCAST, &broadcastEnable, sizeof(broadcastEnable));
                memset(&udp_addr, 0, sizeof(udp_addr));
                udp_addr.sin_family = AF_INET;
                udp_addr.sin_port = htons(47269);
                udp_addr.sin_addr.s_addr = htonl(INADDR_BROADCAST);
            }
            // Teleplot形式で送信
            char msg[128];
            int msglen = snprintf(msg, sizeof(msg), ">posx:%f\n>posy:%f\n>posz:%f\n", position.x(), position.y(), position.z());
            sendto(udp_sock, msg, msglen, 0, (struct sockaddr*)&udp_addr, sizeof(udp_addr));
        }

        count++;
    }
    
    std::cout << "abort" << std::endl;

    close(fd);
    ENABLE_IMU("0");

    return 0;
}
