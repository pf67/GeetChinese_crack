import Chinese_bboxdetect as bboxdetect
import Chinese_OCR_new as orc
import os
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


def FindImageBBox(img):
    v_sum = np.sum(img, axis=0)
    start_i = None
    end_i = None
    minimun_range = 10
    maximun_range = 20
    min_val = 10
    peek_ranges = []
    ser_val = 0
    # 从左往右扫描，遇到非零像素点就以此为字体的左边界
    for i, val in enumerate(v_sum):
        #定位第一个字体的起始位置
        if val > min_val and start_i is None:
            start_i = i
            ser_val = 0
        #继续扫描到字体，继续往右扫描
        elif val > min_val and start_i is not None:
            ser_val = 0
        #扫描到背景，判断空白长度
        elif val <= min_val  and start_i is not None:
            ser_val  = ser_val + 1
            if (i - start_i >= minimun_range and ser_val > 2) or (i - start_i >= maximun_range):
                # print(i)
                end_i = i
                #print(end_i - start_i)
                if start_i> 5:
                    start_i = start_i-5
                peek_ranges.append((start_i, end_i+2))
                start_i = None
                end_i = None
        #扫描到背景，继续扫描下一个字体
        elif val <= min_val and start_i is None:
            ser_val = ser_val+1
        else:
            raise ValueError("cannot parse this case...")
    return peek_ranges

def getimgnamepath(img_path):
    img_name=[]
    for root, sub_folder, name_list in os.walk(img_path):
        img_name += [os.path.join(root, name) for name in name_list]
    return img_name

def cutgeetcode(main_path,ver_path,img_name_path):
    path = img_name_path.replace('\\','/')
    image = Image.open(path)
    main_box = (0,0,334,343)
    ver_box = (0, 344, 115, 384)
    print(path)
    region = image.crop(main_box)
    main_img_path = main_path + 'main_'+ path.split(r'/')[-1]
    region.save(main_img_path)
    region = image.crop(ver_box)
    ver_img_path = ver_path + 'verificate_'+ path.split(r'/')[-1]
    region.save(ver_img_path)
    return main_img_path,ver_img_path

def run_vercode_boxdetect(ver_img, chineseimg_path):
    ver_img = ver_img.replace('\\', '/')
    image = cv2.imread(ver_img, cv2.IMREAD_GRAYSCALE)
    ret, image1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    box = FindImageBBox(image1)

    imagename = ver_img.split('/')[-1].split('.')[0]
    box_index = 0
    vercode_box_info = []
    cls_vercode_box_info=[]
    bbox = np.array([0,0,0,0])
    for starx,endx in box:
        box_index = box_index +1
        region = image[0:40,starx:endx]
        out_path = chineseimg_path+ imagename +'_'+str(box_index) + '.jpg'
        cv2.imwrite(out_path, region)
        vercode_box_info.append([out_path,bbox])

    cls_vercode_box_info.append(vercode_box_info)
    return cls_vercode_box_info

def JudgeOrcResult(geetcode_bbox_predict_top3,vercode_bbox_predict_top3):
    label_dict = orc.get_label_dict()
    Chinese_Orc_bbox=[]
    for vercode_index, ver_bbox in enumerate(vercode_bbox_predict_top3):
        ver_top3_index = ver_bbox[1]
        entry ={'vercode_index': -1,
                 'predict_index': -1,
                'predict_dict': -1,
                 'predict_box': [0,0,0,0],
                 'predict_success': False}

        entry['vercode_index'] = vercode_index
        entry['predict_success'] = False

        for geetcode_index, geet_bbox in enumerate(geetcode_bbox_predict_top3):
            geet_top3_index = geet_bbox[1]
            for ver_pre_index in ver_top3_index:
                if ver_pre_index in geet_top3_index:
                    entry['predict_index'] = ver_pre_index
                    entry['predict_dict'] = label_dict[str(ver_pre_index)]
                    entry['predict_box'] = geet_bbox[0][1]
                    entry['predict_success'] = True
                    break

            if entry['predict_success'] is True:
                geetcode_bbox_predict_top3.remove(geet_bbox)
                break

        Chinese_Orc_bbox.append(entry)

    for Orc_index, Orc_txt in enumerate(Chinese_Orc_bbox):
        if Orc_txt['predict_success'] is False:
            try:
                geet_bbox = geetcode_bbox_predict_top3[0]
                Orc_txt['predict_box'] = geet_bbox[0][1]
                geetcode_bbox_predict_top3.remove(geet_bbox)
            except:
                print('predict fail')

    return Chinese_Orc_bbox


def show_GeetCodeInImg(imgpath,Chinese_Orc_bbox):
    # Load the test image
    im_file = imgpath.replace('\\', '/')
    im = cv2.imread(im_file)
    # 其中 channel：BGR 存储，而画图时，需要按RGB格式，因此此处作转换。
    im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(12,12),其中参数分别代表子图的行数和列数，一共有 12x12 个图像。函数返回一个figure图像和一个子图ax的array列表。
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i , entry in enumerate(Chinese_Orc_bbox):
        bbox = entry['predict_box']
        class_name = str(entry['predict_dict'])
        index = entry['vercode_index']

        #Matplotlib有一些表示常见图形的对象，这些对象被称为块（patch）。完整的集合位于matplotlib.patches。
        ax.add_patch(
            #长方形 #左下起点，长，宽，颜色等
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        #添加文字
        ax.text(bbox[0], bbox[1] - 2,
                '{:.3f} {:s} '.format(index,class_name),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.show()

if __name__ == '__main__':

    input_path = './data/demo/input/'
    mainimg_path ='./data/demo/main_img/'
    verimg_path ='./data/demo/verificate_img/'
    chineseimg_path = './data/demo/chinese/'
    mainimg_checkpoint_path = './output/chinese/mainimg/'
    verimg_checkpoint_path = './output/chinese/verimg/'


    img_names_path = getimgnamepath(input_path)
    sess1, net1 = bboxdetect.load_faster_rcnn_network()

    chinese_orc_graph_mainimg = tf.Graph()
    sess2, net2 = orc.load_chinese_orc_net(chinese_orc_graph_mainimg,mainimg_checkpoint_path)

    chinese_orc_graph_verimg = tf.Graph()
    sess3, net3 = orc.load_chinese_orc_net(chinese_orc_graph_verimg,verimg_checkpoint_path)

    for img_path in img_names_path:
        main_img,ver_img = cutgeetcode(mainimg_path, verimg_path, img_path)
        geetcode_bbox = bboxdetect.run_geetcode_boxdetect(main_img,chineseimg_path,sess1, net1)
        vercode_bbox = run_vercode_boxdetect(ver_img, chineseimg_path)
        geetcode_bbox_predict_top3 = orc.run_chinese_orc(sess2, net2,geetcode_bbox)
        vercode_bbox_predict_top3 = orc.run_chinese_orc(sess3, net3, vercode_bbox)
        Chinese_Orc_bbox = JudgeOrcResult(geetcode_bbox_predict_top3,vercode_bbox_predict_top3)
        print(Chinese_Orc_bbox)
        show_GeetCodeInImg(img_path,Chinese_Orc_bbox)