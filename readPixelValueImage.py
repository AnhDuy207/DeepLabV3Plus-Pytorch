import cv2

# Đường dẫn tới ảnh
# image_path = '/home/duynguyen/AI/DeepLabV3Plus-Pytorch/datasets/data/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png'
# image_path = '/home/duynguyen/AI/DeepLabV3Plus-Pytorch/testChangePixelValue/aachen_000000_000019_gtFine_labelIds.png'
# image_path = '/home/duynguyen/AI/DeepLabV3Plus-Pytorch/testChangePixelValue/output.png'
image_path = '/home/duynguyen/AI/DeepLabV3Plus-Pytorch/datasets/data/cityscapes_Custom/gtFine/val/lindau/lindau_000027_000019_gtFine_labelIds.png'


# Đọc ảnh bằng OpenCV
image = cv2.imread(image_path)

# Tạo sự kiện chuột
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Hiển thị tọa độ con trỏ chuột
        print(f"Mouse Coordinates: ({x}, {y}) value = ", image[y, x])

# Tạo cửa sổ và gắn callback chuột
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

# Hiển thị ảnh và chờ sự kiện từ người dùng
while True:
    cv2.imshow("Image", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Đóng cửa sổ và giải phóng bộ nhớ
cv2.destroyAllWindows()


# from PIL import Image

# # Đường dẫn tới tệp tin ảnh
# image_path = '/home/duynguyen/AI/DeepLabV3Plus-Pytorch/datasets/data/cityscapes_test/gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png'
# # image_path = '/home/duynguyen/AI/DeepLabV3Plus-Pytorch/testChangePixelValue/aachen_000000_000019_gtFine_labelIds.png'

# # Mở ảnh grayscale
# image = Image.open(image_path).convert("L")

# # Lấy kích thước ảnh
# width, height = image.size

# # Tìm giá trị pixel lớn nhất
# max_pixel = 0
# for y in range(height):
#     for x in range(width):
#         # Lấy giá trị pixel tại vị trí (x, y)
#         pixel_value = image.getpixel((x, y))
#         # print(f"Pixel value at ({x}, {y}): {pixel_value}")
        
#         # Kiểm tra và cập nhật giá trị pixel lớn nhất
#         if pixel_value > max_pixel:
#             max_pixel = pixel_value
#             print(f"Pixel value at ({x}, {y}): {pixel_value}")

# # In giá trị pixel lớn nhất
# print("Giá trị pixel lớn nhất:", max_pixel)