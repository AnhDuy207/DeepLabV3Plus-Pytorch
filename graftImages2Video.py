import os
import cv2

# Đường dẫn đến thư mục chứa ảnh
image_folder = '/home/duynguyen/AI/DeepLabV3Plus-Pytorch/VideoToImages/RealVideoData/output_Predict_Weight_Selfcar'

# Lấy danh sách tất cả các tệp tin trong thư mục
all_images = os.listdir(image_folder)

# Lọc ra các tệp tin ảnh (ví dụ: jpg, png)
image_files = [file for file in all_images if file.lower().endswith('.png')]

# Sắp xếp theo thứ tự tăng dần
sorted_images = sorted(image_files)

# Tạo đối tượng VideoWriter
video_output_path = '/home/duynguyen/AI/DeepLabV3Plus-Pytorch/VideoToImages/RealVideoData/realData_with_selfcar_predict.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Chọn định dạng video
fps = 20  # Số khung hình mỗi giây
width = 540
height = 960
video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

# Lặp qua từng ảnh và thêm vào video
for image in sorted_images:
    image_path = os.path.join(image_folder, image)
    frame = cv2.imread(image_path)
    video_writer.write(frame)

# Giải phóng đối tượng VideoWriter
video_writer.release()

print(f"Video đã được tạo tại: {video_output_path}")