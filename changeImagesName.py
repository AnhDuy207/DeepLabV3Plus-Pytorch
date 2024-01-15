import os

# Thư mục chứa các tệp tin ảnh
folder_path = '/home/duynguyen/AI/DeepLabV3Plus-Pytorch/VideoToImages/RealVideoData/output_Resize_Predcit'

# Lặp qua từng tệp tin trong thư mục
for filename in os.listdir(folder_path):
    
    # Kiểm tra định dạng tệp tin ảnh
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        # Tách phần tên tệp tin và phần mở rộng
        name, ext = os.path.splitext(filename)
        # print(name)
        # print(ext)
        # Kiểm tra nếu phần tên chỉ có hai chữ số
        if len(name) == 8:
            # Đổi tên tệp tin thành ba chữ số
            # Tách phần số từ tên file
            file_number = name.split("_")[1]

            # Định dạng lại số thành chuỗi có 3 chữ số
            formatted_number = file_number.zfill(3)

            # Tạo tên tệp tin mới
            new_filename = "frame_{}.{}".format(formatted_number, ext)
            # Đường dẫn tới tệp tin gốc và tệp tin mới
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            
            # Đổi tên tệp tin
            os.rename(old_path, new_path)

# # Tên tệp tin ảnh gốc
# original_filename = "frame_10.jpg"

# # Tách phần tên file và phần mở rộng
# name, ext = original_filename.split(".")

# # Tách phần số từ tên file
# file_number = name.split("_")[1]

# # Định dạng lại số thành chuỗi có 3 chữ số
# formatted_number = file_number.zfill(3)

# # Tạo tên tệp tin mới
# new_filename = "frame_{}.{}".format(formatted_number, ext)

# print(new_filename)