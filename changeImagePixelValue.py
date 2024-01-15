# from PIL import Image
# import numpy as np

# # Đường dẫn đến tệp hình ảnh
# image_path = "/home/duynguyen/AI/DeepLabV3Plus-Pytorch/testChangePixelValue/aachen_000000_000019_gtFine_labelIds.png"

# # Mở ảnh bằng PIL
# image_pil = Image.open(image_path)

# # Chuyển đổi ảnh PIL thành mảng numpy
# image_array_pil = np.array(image_pil)

# # Đọc ảnh
# image = image_array_pil

# # Lấy kích thước ảnh
# height, width= image.shape

# # In giá trị từng pixel
# for y in range(height):
#     for x in range(width):
#         #road
#         if image[y, x] == 7:
#             image[y, x] = 0
#         #building
#         elif image[y, x] == 11:
#             image[y, x] = 6
#         #tree
#         elif image[y, x] == 21:
#             image[y, x] = 11
#         #sky
#         elif image[y, x] == 23:
#             image[y, x] = 12
#         #car
#         elif image[y, x] == 26:
#             image[y, x] = 4   
#         else:
#             image[y, x] = 1
            
# # Tạo đối tượng Image từ mảng numpy đã chuyển đổi
# image_pil = Image.fromarray(image)

# # Lưu lại ảnh mới
# image_pil.save("/home/duynguyen/AI/DeepLabV3Plus-Pytorch/testChangePixelValue/output.png")



from PIL import Image
import numpy as np
import os

# Thư mục chứa ảnh gốc
input_folder = '/home/duynguyen/AI/DeepLabV3Plus-Pytorch/datasets/data/cityscapes_Custom/gtFine'


# Hàm đệ quy để duyệt qua các thư mục và tệp tin ảnh
def process_folder(folder_path):
    # Lặp qua tất cả các tệp tin và thư mục trong thư mục hiện tại
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        # Kiểm tra nếu là thư mục
        if os.path.isdir(item_path):
            # Nếu là thư mục, tiếp tục đệ quy vào thư mục đó
            process_folder(item_path)
        else:
            # Kiểm tra định dạng tệp tin ảnh
            if item_path.endswith(('labelIds.png')):
                # Đường dẫn tới ảnh gốc
                image_path = os.path.join(input_folder, item_path)
                print(image_path)

                # Mở ảnh bằng PIL
                image_pil = Image.open(image_path)

                # Chuyển đổi ảnh PIL thành mảng numpy
                image_array_pil = np.array(image_pil)

                # Đọc ảnh
                image = image_array_pil

                # Lấy kích thước ảnh
                height, width= image.shape

                # In giá trị từng pixel
                for y in range(height):
                    for x in range(width):
                        #road
                        if image[y, x] == 7:
                            image[y, x] = 0
                        #building
                        elif image[y, x] == 11:
                            image[y, x] = 6
                        #tree
                        elif image[y, x] == 21:
                            image[y, x] = 11
                        #sky
                        elif image[y, x] == 23:
                            image[y, x] = 12
                        #car
                        elif image[y, x] == 26:
                            image[y, x] = 4   
                        else:
                            image[y, x] = 1
                            
                # Tạo đối tượng Image từ mảng numpy đã chuyển đổi
                image_pil = Image.fromarray(image)

                # Lưu lại ảnh mới
                image_pil.save(image_path)

# Gọi hàm đệ quy với thư mục gốc
process_folder(input_folder)