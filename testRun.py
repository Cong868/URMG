from ultralytics import YOLO

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

project=[
    r"/URMG.yaml",
    #ResIR2C analysis
    r"/ResIR2C analysis/Bottleneck.yaml",
    r"/ResIR2C analysis/BottleneckCSP.yaml",
    r"/ResIR2C analysis/C3.yaml",
    r"/ResIR2C analysis/C3K2.yaml",
    r"/ResIR2C analysis/ResIR2C.yaml",
    #Ablation experiment
    r"/Ablation experiment/1.yaml",
    r"/ResIR2C analysis/2.yaml",
    r"/ResIR2C analysis/3.yaml",
]
if __name__=='__main__':
    for i in range(len(project)):
        model = YOLO(project[i])  # build a new model from scratch
        model.train(data=r"E:\desktop\dataset\visDroneYOLO\data.yaml",epochs=300,imgsz=640, batch=3)  # train the model
        # metrics = model.val(data=r"E:\desktop\dataset\visDroneYOLO\data.yaml")  # evaluatWWe model performance on the validation set
