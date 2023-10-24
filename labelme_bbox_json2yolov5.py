"""
desc: gather json files annotated by labelme into a dictionary,
    and use this script to generate a voc style dataset.

reference: https://github.com/wkentaro/labelme/blob/main/examples/bbox_detection/labelme2voc.py
"""

# coding=utf-8

from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import re
import sys
from progressbar import ProgressBar
import labelme
import imgviz
import shutil
import cv2
import random
import xml.etree.ElementTree as ET

try:
    import lxml.builder
    import lxml.etree
except ImportError:
    print("Please install lxml:\n\n    pip install lxml\n")
    sys.exit(1)



import os.path as osp

def get_label_conversion_dict(dict_file):
    """
    自定义标签转换，例如将中文标签转换为英文标签
    custom label conversion, for example, convert chinese label to english label, vice versa.
    """
    if dict_file is None:
        return {}
    with open(dict_file, "r", encoding='UTF-8') as dict_f:
        label_dict = {}
        for line in dict_f:
            line = line.strip()
            if line == "":
                continue
            words = line.split(":")
            label_dict[words[0].strip()] = words[1].strip()
    return label_dict


def get_coco_category(labels_file):
    """生成标签字典，用于生成COCO数据集时供查询"""
    if not osp.exists(labels_file):
        print('file not exists:', labels_file)
        return None
    attr_dict = {"categories": []}
    label_id = 0
    with open(labels_file, "r", encoding='UTF-8') as label_f:
        for line in label_f:
            label = line.strip()
            label_item = {"supercategory": "defect", "id": label_id, "name": label}
            attr_dict["categories"].append(label_item)
            label_id += 1
    return attr_dict

# regex for get base name
pattern = re.compile(r"\d+")  # e.g. "擦花20180830172530对照样本.jpg" -> "20180830172530"


def get_base_name(file_name):
    """
    get base name per json file
    TODO: define the way generate base name.
    """
    # 1. use regex get item name

    # filename = osp.splitext(osp.basename(file_name))[0]
    # base = pattern.findall(filename)[0]

    # 2. just use original filename
    base = osp.splitext(osp.basename(file_name))[0]
    return base


def process_labels(label_file, label_dict, out_class_names_file):
    """get labels and save it to dataset dir"""
    class_names = []
    with open(label_file, "r", encoding="UTF-8") as label_f:
        for i, line in enumerate(label_f.readlines()):
            class_id = i - 1  # starts with -1
            class_name = line.strip()
            if class_id == -1:
                assert class_name == "__ignore__"
                continue
            if class_id == 0:
                assert class_name == "_background_"
            if class_name in label_dict:
                class_name = label_dict[class_name]
            class_names.append(class_name)

        class_names = tuple(class_names)
        print("class_names:", class_names)
        # save labels in txt for information
        with open(out_class_names_file, "w", encoding="UTF-8") as out_f:
            out_f.writelines("\n".join(class_names))
        print("Saved class_names:", out_class_names_file)
    return class_names


def get_bbox_boundaries(shape):
    """get box points
    TODO: define the way calculate box four point here.
    """
    # MARK: Please Confirm the box format in your dataset.
    # ⚠️：大家确认下自己使用的数据中 获取BBOX的方式正不正确。是否需要调整下标。

    # (xmin, ymin), (xmax, ymax) = shape["points"]

    xmin = shape['points'][0][0]
    ymin = shape['points'][0][1]
    xmax = shape['points'][1][0]
    ymax = shape['points'][1][1]
    # swap if min is larger than max.
    xmin, xmax = sorted([xmin, xmax])
    ymin, ymax = sorted([ymin, ymax])
    # be care of the difference between your dataset image Coordinate and labelme imgViz Coordinate.

    # return (xmin, ymin, xmax, ymax)
    return ymin, xmin, ymax, xmax


def get_basic_maker_and_xml(shape, filename):
    """get basic maker"""
    maker = lxml.builder.ElementMaker()
    maker_size = maker.size(
            maker.height(str(shape[0])),
            maker.width(str(shape[1])),
            maker.depth(str(shape[2])),
    )
    maker_source = maker.source(
            maker.database(""),
            maker.annotation(""),
            maker.image(""),
    )
    xml = maker.annotation(
        # folder name
        maker.folder(""),
        # img path
        maker.filename(filename),
        # img source, ignore it
        maker_source,
        # image size(height, width and channel)
        maker_size,
        # add category if it's for segmentation
        maker.segmented("0"),
    )
    return maker, xml


def append_bbox_to_xml(maker, xml, box, class_name):
    """append bbox to xml"""
    # object info
    maker_bndbox = maker.bndbox(
                    maker.xmin(str(box[1])),
                    maker.ymin(str(box[0])),
                    maker.xmax(str(box[3])),
                    maker.ymax(str(box[2])),
    )
    bbox_obj = maker.object(
                # label name
                maker.name(class_name),
                # pose info, ignore
                maker.pose(""),
                # truncated info, ignore
                maker.truncated("0"),
                # difficulty, ignore
                maker.difficult("0"),
                # bbox(up-left corner and bottom-right corner points)
                maker_bndbox,
    )
    xml.append(bbox_obj)
    return xml


def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def get_xml_with_labelfile(label_file, base, label_dict, class_names):
    """
    get_xml_with_labelfile
    @param label_file:
    @param base:
    @param label_dict:
    @param class_names:
    @return:
    """
    img = labelme.utils.img_data_to_arr(label_file.imageData)

    # generate voc format annotation file
    (maker, xml) = get_basic_maker_and_xml(img.shape, base + ".jpg")

    # two list for visualization
    bboxes = []
    labels = []
    # MARK: change it for annotation shape type, some use points, some use rectangle.
    # Here shows the points one.
    for shape in label_file.shapes:
        box = get_bbox_boundaries(shape=shape)
        if box is None:
            continue

        class_name = shape["label"]  # object name in json file
        if class_name in label_dict:
            class_name = label_dict[class_name]
            class_id = class_names.index(class_name)  # convert to class id
        else:
            class_name = "Tea"
            class_id = 1

        bboxes.append([box[0], box[1], box[2], box[3]])
        labels.append(class_id)

        xml = append_bbox_to_xml(maker, xml, box, class_name)

    return xml, bboxes, labels


def process_annotated_json(class_names, filename, output_dir, label_dict):
    """translate to image and xml"""
    # file nam base
    base = get_base_name(filename)
    # src image file
    out_img_file = osp.join(output_dir, "images", base + ".jpg")
    # annotation xml file
    out_xml_file = osp.join(output_dir, "Annotations", base + ".xml")
    # viz image file
    out_viz_file = osp.join(output_dir, "AnnotationsVisualization", base + ".jpg")

    label_file = labelme.LabelFile(filename=filename)

    # save source image
    img = labelme.utils.img_data_to_arr(label_file.imageData)
    imgviz.io.imsave(out_img_file, img)

    # get xml
    (xml, bboxes, labels) = get_xml_with_labelfile(label_file, base, label_dict, class_names)

    # save visualized image
    save_visualization_image(img, labels, bboxes, class_names, output_file=out_viz_file)

    # save voc annotation to xml file
    with open(out_xml_file, "wb") as out_f:
        out_f.write(lxml.etree.tostring(xml, pretty_print=True))


def save_visualization_image(img, labels, bboxes, class_names, output_file):
    """save visualized image"""
    # caption for visualize drawing
    captions = [class_names[label] for label in labels]
    viz = imgviz.instances2rgb(
        image=img,
        labels=labels,
        bboxes=bboxes,
        captions=captions,
        font_size=15,
    )
    imgviz.io.imsave(output_file, viz)

def convert_label(path, image_id, classes):
    def convert_box(size, box):
        dw = 1. / (size[0])
        dh = 1. / (size[1])
        x = (box[0] + box[1]) / 2.0 - 1
        y = (box[2] + box[3]) / 2.0 - 1
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h
 
    in_file = open(os.path.join(path,f'Annotations/{image_id}.xml'),encoding='utf-8')
    out_file = open(os.path.join(path, f'labels/{image_id}.txt'), 'w',encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
 
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls in classes and not int(obj.find('difficult').text) == 1:
            xmlbox = obj.find('bndbox')
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
            cls_id = classes.index(cls)  # class id
            out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')

def main():
    """main"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--json_dir", default='D:\\MT3D\\fall', help="input annotated directory")
    parser.add_argument("--output_dir", default='D:\\MT3D_VOC\\fall', help="output dataset directory")
    parser.add_argument("--labels", default='D:\\MT3D\\fall\\label_names.txt', help="labels file")
    parser.add_argument("--label_dict", default= 'D:\\MT3D\\fall\\label_dict.txt', help="convert label with dict")
    args = parser.parse_args()

    # make voc format directory
    if osp.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "images"))
    os.makedirs(osp.join(args.output_dir, "Annotations"))
    os.makedirs(osp.join(args.output_dir, "AnnotationsVisualization"))
    print("Creating dataset:", args.output_dir)

    label_file_list = glob.glob(osp.join(args.json_dir, "*.json"))

    # build label conversion dict
    fst2snd_dict = get_label_conversion_dict(args.label_dict)

    # get labels and save it to dataset dir
    out_class_names_file = osp.join(args.output_dir, "classes.txt")
    class_names = process_labels(label_file=args.labels,
                                 label_dict=fst2snd_dict,
                                 out_class_names_file = out_class_names_file)
    # 遍历处理
    pbar = ProgressBar().start()
    pbar.maxval = len(label_file_list)
    for i, filename in enumerate(label_file_list):
        process_annotated_json(class_names=class_names,
                               filename=filename,
                               output_dir=args.output_dir,
                               label_dict=fst2snd_dict)
        pbar.update(i + 1)
    pbar.finish()

    bg_img_file_list = []
    t_bg_img_file_list = glob.glob(osp.join(args.json_dir, "*.png"))
    pbar1 = ProgressBar().start()
    pbar1.maxval = len(t_bg_img_file_list)
    for i, filename in enumerate(t_bg_img_file_list):
        img = cv2.imread(filename)
        if img is None:
            continue
        filename = os.path.basename(filename)
        filename = filename[:36] + ".jpg"
        out_img_file = osp.join(args.output_dir, "images", filename)
        cv2.imwrite(out_img_file,img)
        bg_img_file_list.append(out_img_file)
        pbar1.update(i + 1)
    pbar1.finish()

    trainval_percent = 0.8
    train_percent = 0.8

    classes, _ = get_classes(out_class_names_file)
    annotation_mode = 0

    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate train/test/val txt")
        xmlfilepath = os.path.join(args.output_dir, 'Annotations')
        saveBasePath = args.output_dir
        temp_xml = os.listdir(xmlfilepath)
        total_xml = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)
 
        num = len(total_xml)
        list = [i for i in range(num)]
        tv = int(num * trainval_percent)
        tr = int(tv * train_percent)
        trainval = random.sample(list, tv)
        train = random.sample(trainval, tr)
        random.shuffle(list)
 
        print("train and val size", tv)
        print("train size", tr)
        ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w',encoding='utf-8')
        ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w',encoding='utf-8')
        ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w',encoding='utf-8')
        fval = open(os.path.join(saveBasePath, 'val.txt'), 'w',encoding='utf-8')

        dev_path = args.output_dir
        for i in list:
            name = total_xml[i][:-4]
            if i in trainval:
                ftrainval.write('%s/images/%s.jpg\n'% (dev_path, name))
                if i in train:
                    ftrain.write('%s/images/%s.jpg\n'% (dev_path, name))
                else:
                    fval.write('%s/images/%s.jpg\n'% (dev_path, name))
            else:
                ftest.write('%s/images/%s.jpg\n'% (dev_path, name))
        for path in bg_img_file_list:
            ftrain.write(path+"\n")
            ftrainval.write(path+"\n")
 
        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest.close()
        print("Generate train/test/val.txt done.")
 
    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate coco labels")
        if not os.path.exists(os.path.join(args.output_dir,'labels/')):
            os.makedirs(os.path.join(args.output_dir,'labels/'))
        image_ids = os.listdir(os.path.join(args.output_dir, 'Annotations'))
        for image_id in image_ids:
            convert_label(args.output_dir, image_id[:-4], classes=classes)
        print("Generate YOLO labels done")

if __name__ == "__main__":
    main()
