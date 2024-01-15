import json
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import os
import io

# def create_label_image_from_json(json_data, output_path):
#     # Extract image size
#     image_size = (json_data["size"]["width"], json_data["size"]["height"])

#     # Initialize an array for label data
#     label_data = np.zeros((image_size[1], image_size[0], 4), dtype=np.uint8)

#     # Process each object in the JSON data
#     for obj in json_data["objects"]:
#         class_id = obj["classId"]
#         class_title = obj["classTitle"]
#         bitmap_data = obj["bitmap"]["data"]
#         origin = obj["bitmap"]["origin"]
#         #rint(f'data: {type(bitmap_data)}')

#         # Decode base64 string to bytes
#         bitmap_bytes = base64.b64decode(bitmap_data)
#         print(f'bytes: {bitmap_bytes}')
#         #point = struct.unpack('<f', bitmap_bytes)
#         #print(f'point: {point}')
#         print(f'bytes: {(bitmap_bytes)}')
#         print('1')
#         print(f'bytesio: {BytesIO(bitmap_bytes)}')
#         #img = Image.frombytes("RGBA", (image_size[1], image_size[0]), bitmap_bytes)
#         img = Image.frombytes("RGBA", image_size, bitmap_bytes)

#         img.show()
#         # Create an image from the bytes
#         if bitmap_bytes[:8] == b'\x89PNG\r\n\x1a\n':
#             try:
#              #   with Image.open(BytesIO('bitmap')) as bitmap_image:
#                 # bitmap_array = np.asarray(bytearray(bitmap_bytes, dtype=np.uint8))
#                 # bitmap_image = Image.fromarray(bitmap_array)
#                 # # Resize the image to the specified size
#                 # bitmap_image = bitmap_image.resize((image_size[0], image_size[1]))
#                 # bitmap_array = np.array(bitmap_image)
#                 bitmap_array = np.frombuffer(bitmap_bytes, dtype=np.uint8)

#                 # Reshape the array to the specified image size
#                 bitmap_array = bitmap_array[:image_size[0] * image_size[1] * 4]  # Ensure correct length
#                 bitmap_array = bitmap_array.reshape((image_size[1], image_size[0], 4))
#         # Place the bitmap data onto the label array at the specified origin
#                 label_data[
#                     origin[1]:origin[1] + bitmap_array.shape[0],
#                     origin[0]:origin[0] + bitmap_array.shape[1],
#                     :3
#                 ] = bitmap_array[:, :, :3]  # Exclude the alpha channel
#                 label_data[
#                     origin[1]:origin[1] + bitmap_array.shape[0],
#                     origin[0]:origin[0] + bitmap_array.shape[1],
#                     3
#                 ] = class_id  # Set the alpha channel to the class ID
#             except Exception as e:
#                 print(f"Error processing object {obj['id']}: {e}")
#         else:
#             print(f"Error: Object {obj['id']} does not contain a valid PNG image.")

#     # Convert the label data array to an image
#     label_image = Image.fromarray(label_data)

#     # Save the label image
#     label_image = Image.fromarray(label_data, mode="L")
#     label_image.save(output_path)

def create_label_image_from_json(json_data, output_path):
    # Trích xuất kích thước ảnh từ dữ liệu JSON
    height = json_data["size"]["height"]
    width = json_data["size"]["width"]

    # Khởi tạo mảng label_data
    label_data = np.zeros((height, width), dtype=np.uint8)

    # Xử lý từng đối tượng trong dữ liệu JSON
    for obj in json_data["objects"]:
        # Trích xuất thông tin từ đối tượng
        class_id = obj["classId"]
        class_title = obj["classTitle"]
        bitmap_data = obj["bitmap"]["data"]
        origin = obj["bitmap"]["origin"]

        print("before -> bitmap_data = ", bitmap_data)
        # split 1 doan text mong muon
        # bitmap_data = bitmap_data.split("///")[1]
        # print("after -> bitmap_data = ", bitmap_data)
        
        # Giải mã chuỗi base64 thành dữ liệu byte
        decoded_data = base64.b64decode(bitmap_data)
        print("decoded_data = ", decoded_data)
        
        # Tạo hình ảnh từ dữ liệu byte
        image = Image.open(io.BytesIO(decoded_data)).convert("RGBA")
        
        # try:
        #     image = Image.open(io.BytesIO(decoded_data)).convert("RGBA")
        # except Exception as e:
        #     print(f"Error decoding image data: {str(e)}")
        #     continue

        # Chuyển đổi hình ảnh thành mảng numpy
        bitmap_array = np.array(image)

        # Đặt dữ liệu bitmap lên mảng label_data tại vị trí gốc đã xác định
        x_start = origin[0]
        y_start = origin[1]
        label_data[y_start:y_start+bitmap_array.shape[0], x_start:x_start+bitmap_array.shape[1]] = bitmap_array[:,:,3]

    # Chuyển đổi mảng label_data thành hình ảnh label
    label_image = Image.fromarray(label_data)

    # Lưu hình ảnh label vào đường dẫn đầu ra
    label_image.save(output_path)

if __name__ == "__main__":
    # Replace "your_data.json" with the path to your JSON file
    json_file = "/home/duynguyen/AI/DeepLabV3Plus-Pytorch/datasets/data/selfcar/ann/train/02_00_000.png.json"
    output_image_path = "/home/duynguyen/AI/DeepLabV3Plus-Pytorch/datasets/data/selfcar/labels"  # Replace with the desired output path
    with open(json_file, 'r') as f:
        data = json.load(f)
    json_file_name = os.path.basename(json_file)
    output_file_name = os.path.splitext(json_file_name)[0] + "_labels.png"

    # Combine the folder path and the generated file name
    output_image_path = os.path.join(output_image_path, output_file_name)

    create_label_image_from_json(data, output_image_path)


