import cv2

# Đường dẫn đến file video
video_path = "/home/duynguyen/AI/DeepLabV3Plus-Pytorch/VideoToImages/RealVideoData/realVideoData.MOV"
output_folder = "/home/duynguyen/AI/DeepLabV3Plus-Pytorch/VideoToImages/RealVideoData/output/"

# Khởi tạo đối tượng VideoCapture
cap = cv2.VideoCapture(video_path)

# Kiểm tra xem video có mở thành công hay không
if not cap.isOpened():
    print("Không thể mở video.")
    exit()

# Lấy tỷ lệ khung hình của video
frame_rate = cap.get(cv2.CAP_PROP_FPS)
# print("frame_rate = ", frame_rate)

# Số lượng frame cần tách trên 1s
frames_per_second = 19  # Thay đổi giá trị này cho số lượng frame mong muốn

# Số frame cần tách trong tổng số frame video
frames_to_extract = int(frame_rate / frames_per_second)

frame_count = 0

frame_num = 0

# Đọc từng khung hình trong video và lưu thành các file hình ảnh
while True:
    # Đọc từng khung hình
    ret, frame = cap.read()

    # Kiểm tra nếu không còn khung hình nào
    if not ret:
        break
    
    # Kiểm tra xem frame_count có phải là frame cần tách hay không
    if frame_count % frames_to_extract == 0:
        # print("frames_to_extract = ", frames_to_extract)
        # Tạo tên file cho khung hình
        frame_name = "frame_" + str(frame_num) + ".jpg"

        # Lưu frame vào thư mục đầu ra
        output_path = output_folder + frame_name
        cv2.imwrite(output_path, frame)
        frame_num += 1

    # Tăng biến đếm số khung hình
    frame_count += 1

    # Hiển thị số khung hình đang xử lý
    print("Đã tách khung hình", frame_count)

# Giải phóng đối tượng VideoCapture và kết thúc chương trình
print("Split DONE")
cap.release()
cv2.destroyAllWindows()





