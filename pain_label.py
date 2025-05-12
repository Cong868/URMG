# 多标签 jk
# 用于红外小目标标签展示


labelsource = r'/home/yangcong/下载/open-sirst-v2-master/labels'
import os
import cv2
def convert(x, y,w,h,size):
    x_center=x
    y_center = y
    return (int((x_center-w/2)*size[1]), int((y_center -h/2)*size[0]), int((x_center+w/2)*size[1]), int((y_center+w/2)*size[0]))
dir_list =  os.listdir("/home/yangcong/下载/open-sirst-v2-master/labels")
image_path=r"/home/yangcong/下载/open-sirst-v2-master/targets"

# # label_namne={"0":"pedestrian",
# #              "1":"people",
# #              "2":"bicycle",
# #              "3":"car",
# #              "4":"van",
# #              "5":"truck",
# #              "6":"tricycle",
# #              "7":"awning-tricycle",
# #              "8":"bus",
# #              "9":"motor",}
# # label_box_color={"0":(255,42,4),
# #                  "1":(235,219,11),
# #                  "2":(243,243,243),
# #                  "3":(183,223,0),
# #                  "4":(104,31,17),
# #                  "5":(221,111,255),
# #                  "6":(79,68,255),
# #                  "7":(0,237,204),
# #                  "8":(68,243,0),
# #                  "9":(255,0,189)}





# # label_namne={"0":"person",
# #              "1":"car",
# #              "2":"cyclist",
# #              "3":"truck",
# #              "4":"bike",
# #              "5":"bus"}
# # label_box_color={"0":(255,42,4),
# #                  "1":(235,219,11),
# #                  "2":(243,243,243),
# #                  "3":(183,223,0),
# #                  "4":(104,31,17),
# #                  "5":(221,111,255)}
label_namne={"0":"airplane"}
label_box_color={"0":(255,42,4)}



# label_namne={"0":"person",
#              "1":"car",
#              "2":"cyclist",
#              "3":"truck",
#              "4":"bike",
#              "5":"bus"}
# label_box_color={"0":(255,42,4),labelsource = r'/home/yangcong/下载/sirst-master/labels'
# import os
# import cv2
# def convert(x, y,w,h,size):
#     x_center=x
#     y_center = y
#     return (int((x_center-w/2)*size[1]), int((y_center -h/2)*size[0]), int((x_center+w/2)*size[1]), int((y_center+w/2)*size[0]))
# dir_list =  ["Misc_1.txt","Misc_2.txt"]
# image_path=r"/home/yangcong/下载/sirst-master/images/images"

# # label_namne={"0":"pedestrian",
# #              "1":"people",
# #              "2":"bicycle",
# #              "3":"car",
# #              "4":"van",
# #              "5":"truck",
# #              "6":"tricycle",
# #              "7":"awning-tricycle",
# #              "8":"bus",
# #              "9":"motor",}
# # label_box_color={"0":(255,42,4),
# #                  "1":(235,219,11),
# #                  "2":(243,243,243),
# #                  "3":(183,223,0),
# #                  "4":(104,31,17),
# #                  "5":(221,111,255),
# #                  "6":(79,68,255),
# #                  "7":(0,237,204),
# #                  "8":(68,243,0),
# #                  "9":(255,0,189)}





# # label_namne={"0":"person",
# #              "1":"car",
# #              "2":"cyclist",
# #              "3":"truck",
# #              "4":"bike",
# #              "5":"bus"}
# # label_box_color={"0":(255,42,4),
# #                  "1":(235,219,11),
# #                  "2":(243,243,243),
# #                  "3":(183,223,0),
# #                  "4":(104,31,17),
# #                  "5":(221,111,255)}
# # label_namne={"0":"airplane"
# #              }
# # label_box_color={"0":(255,42,4)}



# label_namne={"0":"person",
#              "1":"car",
#              "2":"cyclist",
#              "3":"truck",
#              "4":"bike",
#              "5":"bus"}
# label_box_color={"0":(255,42,4),
#                  "1":(235,219,11),
#                  "2":(243,243,243),
#                  "3":(183,223,0),
#                  "4":(104,31,17),
#                  "5":(221,111,255)}
# label_box_color={"0":(255,42,4)}
# i=0
# for filename in dir_list:
#     with open(os.path.join(labelsource,filename)) as file:
#         filename_prefix = filename.split(".")[0]
#         print(os.path.join(image_path, f"{filename_prefix}.png"))
#         img = cv2.imread(os.path.join(image_path, f"{filename_prefix}.png"))
#         for line in file:
#             params = line.strip().split(' ')
#             x1,y1,x2,y2=convert(float(params[1]),float(params[2]),float(params[3]),float(params[4]),img.shape)
#             classtype = params[0]
#             if classtype=="0":
#                 print(label_box_color[classtype])
#                 print(label_namne[classtype])
#             p1,p2=(x1,y1),(x2,y2)
#             img=cv2.rectangle(img,(x1,y1),(x2,y2),color=label_box_color["0"],thickness=2)
#             w, h = cv2.getTextSize(label_namne[classtype], 0, fontScale=0.66666, thickness=1)[0]  # text width, height
#             outside = p1[1] - h >= 3
#             p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
#             # img=cv2.rectangle(img, p1, p2, label_box_color[classtype], -1, cv2.LINE_AA)  # filled
#             # img=cv2.putText(img,label_namne[classtype],color=(255,255,255),org=(p1[0], p1[1] - 4 if outside else p1[1] + h + 2),fontFace=0,fontScale=0.66666,thickness=1,lineType=cv2.LINE_AA)
#         cv2.imwrite(f"./output{i}.jpg",img)
#         i+=1
#                  "1":(235,219,11),
#                  "2":(243,243,243),
#                  "3":(183,223,0),
#                  "4":(104,31,17),
#                  "5":(221,111,255)}
label_box_color={"0":(255,42,4)}
i=0
for filename in dir_list:
    with open(os.path.join(labelsource,filename)) as file:
        filename_prefix = filename.split(".")[0].replace("_pixels0","")
        print(os.path.join(image_path, f"{filename_prefix}.png"))
        print(os.path.join(image_path, f"{filename_prefix}.png"))
        img = cv2.imread(os.path.join(image_path, f"{filename_prefix}.png"))
        print(img)
        for line in file:
            params = line.strip().split(' ')
            x1,y1,x2,y2=convert(float(params[1]),float(params[2]),float(params[3]),float(params[4]),img.shape)
            classtype = params[0]
            if classtype=="0":
                print(label_box_color[classtype])
                print(label_namne[classtype])
            p1,p2=(x1,y1),(x2,y2)
            img=cv2.rectangle(img,(x1,y1),(x2,y2),color=label_box_color["0"],thickness=2)
            w, h = cv2.getTextSize(label_namne[classtype], 0, fontScale=0.66666, thickness=1)[0]  # text width, height
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            # img=cv2.rectangle(img, p1, p2, label_box_color[classtype], -1, cv2.LINE_AA)  # filled
            # img=cv2.putText(img,label_namne[classtype],color=(255,255,255),org=(p1[0], p1[1] - 4 if outside else p1[1] + h + 2),fontFace=0,fontScale=0.66666,thickness=1,lineType=cv2.LINE_AA)
        cv2.imwrite(f"/home/yangcong/下载/open-sirst-v2-master/wan/output{i}.jpg",img)
        i=i+1
#
#         i+=1
# import cv2
# img=cv2.imread("/home/yangcong/下载/sirst/test/images/S20210713_S3_3.png")
# img=cv2.rectangle(img,(293,249),(307,258),(255,255,0),1)
# cv2.imshow("wan",img)
# cv2.waitKey(0)
# cv2.destroyWindow()





 

# import xml.etree.ElementTree as ET
# import os
# from pathlib import Path
# def convert(size, box):
#     x_center = (box[0] + box[1]) / 2.0
#     y_center = (box[2] + box[3]) / 2.0
#     x = x_center / size[0]
#     y = y_center / size[1]
#     w = (box[1] - box[0]) / size[0]
#     h = (box[3] - box[2]) / size[1]
#     return (x, y, w, h)
# wan=os.listdir(r"/home/yangcong/下载/sirst-master/masks/masks")
# for i in wan:
#     print(i)
#     fix=i.split(".")[-1]
#     finename=i.split(".")[0]
#     if fix=="xml":
#         if finename == "":
#             continue
#         tree = ET.parse(os.path.join(r"/home/yangcong/下载/sirst-master/masks/masks", f"{finename}.xml"))
#         root = tree.getroot()
#         size = root.find('size')
#         img_width = int(size.find('width').text)
#         img_height = int(size.find('height').text)
#         with open(os.path.join("/home/yangcong/下载/sirst-master/labels", f"{finename}.txt"), "w+") as f:
#             for obj in root.findall('object'):
#                 xmin = int(obj.find("bndbox").find("xmin").text)
#                 ymin = int(obj.find("bndbox").find("ymin").text)
#                 xmax = int(obj.find("bndbox").find("xmax").text)
#                 ymax = int(obj.find("bndbox").find("ymax").text)
#                 x, y, w, h = convert((img_width, img_height), (xmin, xmax, ymin, ymax))
#                 f.write(f"0 {x} {y} {w} {h}\n")