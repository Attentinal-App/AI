// #include <dlib/image_loader/image_loader.h>
// #include <dlib/image_transforms.h>
// #include <dlib/opencv.h>
// #include <dlib/image_processing/frontal_face_detector.h>
// #include <dlib/image_processing/render_face_detections.h>
// #include <dlib/gui_widgets.h>
// #include <opencv2/opencv.hpp>

// int main() {
//     // 얼굴 검출기 초기화
//     dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

//     // 이미지를 불러옵니다.
//     dlib::array2d<dlib::rgb_pixel> img;
//     dlib::load_image(img, "2007_007763.jpg"); // 사용할 이미지 파일 경로

//     // 얼굴 검출
//     std::vector<dlib::rectangle> dets = detector(img);
    
//     // 결과 표시
//     dlib::image_window win;
//     win.set_image(img);
//     win.add_overlay(dets, dlib::rgb_pixel(255, 0, 0)); // 얼굴 영역 표시

//     win.wait_until_closed();
//     return 0;
// }

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/gui_widgets.h>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>

int main() {
    // 얼굴 검출기 초기화
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

    // OpenCV를 사용하여 이미지를 불러옵니다.
    cv::Mat img = cv::imread("your_image.jpg"); // 이미지 파일 경로
    if (img.empty()) {
        std::cerr << "이미지를 불러올 수 없습니다." << std::endl;
        return 1;
    }

    // dlib 포맷으로 변환
    dlib::cv_image<dlib::bgr_pixel> dlib_img(img);

    // 얼굴 검출
    std::vector<dlib::rectangle> dets = detector(dlib_img);
    
    // 결과 표시
    dlib::image_window win;
    win.set_image(dlib_img);
    win.add_overlay(dets, dlib::rgb_pixel(255, 0, 0)); // 얼굴 영역 표시

    win.wait_until_closed();
    return 0;
}
