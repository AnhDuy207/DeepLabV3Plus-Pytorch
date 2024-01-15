# # Code for one Image
# import cv2

# # Đường dẫn tới ảnh gốc
# image_path = '/home/duynguyen/AI/DeepLabV3Plus-Pytorch/VideoToImages/RealVideoData/output/frame_160.jpg'

# # Đọc ảnh bằng OpenCV
# image = cv2.imread(image_path)

# # Kích thước mới
# new_size = (540, 960)  # Thay đổi width và height theo ý muốn

# # Thay đổi kích thước ảnh
# resized_image = cv2.resize(image, new_size)

# # Lưu ảnh mới
# cv2.imwrite('/home/duynguyen/AI/DeepLabV3Plus-Pytorch/test_results/frame_160_resize.jpg', resized_image)

# Code for Images folder
import cv2
import os

# Thư mục chứa ảnh gốc
input_folder = '/home/duynguyen/AI/DeepLabV3Plus-Pytorch/VideoToImages/RealVideoData/output'

# Thư mục lưu ảnh đã thay đổi kích thước
output_folder = '/home/duynguyen/AI/DeepLabV3Plus-Pytorch/VideoToImages/RealVideoData/output_Resize'

# Kích thước mới
new_size = (540, 960)  # Thay đổi width và height theo ý muốn

# Lặp qua từng tệp tin ảnh trong thư mục đầu vào
for filename in os.listdir(input_folder):
    # Kiểm tra định dạng tệp tin ảnh
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        # Đường dẫn tới ảnh gốc
        image_path = os.path.join(input_folder, filename)

        # Đọc ảnh bằng OpenCV
        image = cv2.imread(image_path)

        # Thay đổi kích thước ảnh
        resized_image = cv2.resize(image, new_size)

        # Đường dẫn tới ảnh đã thay đổi kích thước
        output_path = os.path.join(output_folder, filename)

        # Lưu ảnh đã thay đổi kích thước
        cv2.imwrite(output_path, resized_image)