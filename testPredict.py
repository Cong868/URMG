from ultralytics import YOLO
#程序中链接了多个 OpenMP 运行时库的副本
import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
start_time = time.time()  # 开始时间
# 这里调用YOLO模型对图像进行处理
model = YOLO(r"E:\desktop\paper\run_file\VisDrone\yolov8n\weights\best.pt")
# model.predict(r"E:\desktop\dataset\visDroneYOLO\VisDrone2019-DET-val\images")
result=model.predict(r"E:\desktop\dataset\visDroneYOLO\VisDrone2019-DET-val\images\0000287_02601_d_0000772.jpg")
print(result)
result[0].save("./result.jpg")
end_time = time.time()    # 结束时间
inference_time = end_time - start_time  # 推理时间
print(inference_time)

