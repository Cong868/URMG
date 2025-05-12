from ultralytics import YOLO


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

project=[
    # r"/home/yangcong/URMG-main/ultralytics/cfg/models/v8/yolov8.yaml",

    # #ResIR2C analysis
    r"/home/yangcong/下载/best.pt",
    # r"/ResIR2C analysis/BottleneckCSP.yaml",
    # r"/ResIR2C analysis/C3.yaml",Y
    # r"/ResIR2C analysis/C3K2.yaml",
    # r"/ResIR2C analysis/ResIR2C.yaml",
    # #Ablation experimenYO
    # r"/Ablation experiment/1.yaml",
    # r"/ResIR2C analysis/2.yaml",
    # r"/ResIR2C analysis/3.yaml", 
]
if __name__=='__main__':
    for i in range(len(project)):
        model = YOLO(project[i])  # build a new model from scratch
        # model.train(data=r"/home/yangcong/visDroneYOLO/data.yaml",epochs=300,imgsz=640, batch=2)  # train the model
        metrics = model.val(data=r"/home/yangcong/visDroneYOLO/data.yaml")
        # print(metrics)# evaluatWWe model performance on the validation set
