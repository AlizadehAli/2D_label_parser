import os
from os import walk, getcwd
import json
from typing import List, Any
from PIL import Image
from tqdm import tqdm

import argparse


__author__ = "Ali Alizadeh"
__email__ = 'aalizade@ford.com.tr / a.alizade@live.com'
__license__ = 'AA_Parser'

def parse_arguments():
    parser = argparse.ArgumentParser(description= 'YOLOv3 json, Berkeley Deep Drive dataset (BDD100K), nuscenes 2D labels to txt-label format for yolov3 darknet NN model')
    parser.add_argument("-dt", "--data_type",
                        default="yolo",
                        help="data type of interest; yolo, bdd, nuscenes")
    parser.add_argument("-l", "--label_dir", default="./labels/",
                        help="root directory of the labels for YOLO json file, Berkeley Deep Drive (BDD) json-file, nuscenes")
    parser.add_argument("-s", "--save_dir", default="./target_labels/",
                        help="path directory to save the the converted label files")
    parser.add_argument("-i", "--image_dir",
                        default=None, required=False,
                        help="path where the images are located to BDD100K, nescenes, etc.")
    parser.add_argument("-o", "--output_dir",
                        default=None, required=False,
                        help="output directory to save the manipulated image files")
    args = parser.parse_args()
    return args

"""Assistive functions"""
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def collect_bdd_labels(bdd_label_path):
    bdd_json_list = []
    for file in tqdm(os.listdir(bdd_label_path)):
        if file.endswith(".json"):
            bdd_json_list.append(file)
    return bdd_json_list


def sync_labels_imgs(label_path, img_path):
    for path, subdirs, files in tqdm(os.walk(label_path)):
        for file in tqdm(files):
            if file.lower().endswith('.txt'):
                imape_folders_path = img_path + path.split('/')[-1]
                image_path = os.path.join(imape_folders_path, file)
                image_path = image_path.split('.')[0] + '.jpg'
                if not os.path.isdir(image_path):
                    os.remove(image_path)


def write_training_data_path_synced_with_labels(label_path):
    with open('nuscenes_training_dataPath.txt', 'w') as train_data_path:
        for path, subdirs, files in os.walk(label_path):
            for file in files:
                if file.lower().endswith('txt'):
                    full_path = os.path.join(path, file)
                    full_path = full_path.split('.')[0]+ '.jpg'
                    full_path = 'images/'+path.split('/')[-1]+'/'+full_path.split('/')[-1]
                    train_data_path.write(str(full_path) + os.linesep)
        train_data_path.close()

"""Main parser functions"""
def bdd_parser(bdd_label_path):
    bdd_json_list = collect_bdd_labels(bdd_label_path)
    label_data: List[Any] = []
    for file in tqdm(bdd_json_list):
        label_data.append(json.load(open(bdd_label_path+file)))
    return label_data


def yolo_parser(json_path, targat_path):
    json_backup = "./json_backup/"

    wd = getcwd()
    list_file = open('%s_list.txt' % (wd), 'w')

    json_name_list = []
    for file in tqdm(os.listdir(json_path)):
        if file.endswith(".json"):
            json_name_list.append(file)

    """ Process """
    for json_name in tqdm(json_name_list):
        txt_name = json_name.rstrip(".json") + ".txt"
        """ Open input text files """
        txt_path = json_path + json_name
        print("Input:" + txt_path)
        txt_file = open(txt_path, "r")

        """ Open output text files """
        txt_outpath = targat_path + txt_name
        print("Output:" + txt_outpath)
        txt_outfile = open(txt_outpath, "a")

        """ Convert the data to YOLO format """
        lines = txt_file.read().split('\r\n')
        for idx, line in tqdm(enumerate(lines)):
            if ("lineColor" in line):
                break
        if ("label" in line):
            x1 = float(lines[idx + 5].rstrip(','))
            y1 = float(lines[idx + 6])
            x2 = float(lines[idx + 9].rstrip(','))
            y2 = float(lines[idx + 10])
            cls = line[16:17]

            """ in case when labelling, points are not in the right order """
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
    ymax = max(y1, y2)
    img_path = str('%s/dataset/%s.jpg' % (wd, os.path.splitext(json_name)[0]))

    im = Image.open(img_path)
    w = int(im.size[0])
    h = int(im.size[1])

    print(w, h)
    print(xmin, xmax, ymin, ymax)
    b = (xmin, xmax, ymin, ymax)
    bb = convert((w, h), b)
    print(bb)
    txt_outfile.write(cls + " " + " ".join([str(a) for a in bb]) + '\n')


    os.rename(txt_path, json_backup + json_name)  # move json file to backup folder

    """ Save those images with bb into list"""
    if (txt_file.read().count("label") != 0):
        list_file.write('%s/dataset/%s.jpg\n' % (wd, os.path.splitext(txt_name)[0]))

    list_file.close()


def nuscenes_parser(label_path, targat_path, img_path):
    json_backup = "json_backup/"
    wd = getcwd()

    sync_labels_imgs(label_path, img_path)
    """ Get input json file list """
    json_name_list = []
    for file in tqdm(os.listdir(label_path)):
        if file.endswith(".json"):
            json_name_list.append(file)
            data = json.load(open(label_path + file))

            # Aggregate the bounding boxes associate with each image
            unique_img_names = []
            for i in tqdm(range(len(data))):
                unique_img_names.append(data[i]['filename'])

            unique_img_names = list(dict.fromkeys(unique_img_names))

            i: int
            for i in tqdm(range(len(unique_img_names))):
                f = open(targat_path + unique_img_names[i].split('/')[1] + '/' + unique_img_names[i].split('/')[-1].split('.')[0] + '.txt', "w+")
                for idx, name in enumerate(data):
                    if unique_img_names[i] == name['filename']:
                        x, y, w, h = convert((1600, 900), name['bbox_corners'])
                        if 'pedestrian' in name['category_name']:
                            obj_class = 0
                        elif 'bicycle' in name['category_name']:
                            obj_class = 1
                        elif 'motorcycle' in name['category_name']:
                            obj_class = 2
                        elif 'car' in name['category_name']:
                            obj_class = 3
                        elif 'bus' in name['category_name']:
                            obj_class = 4
                        elif 'truck' in name['category_name']:
                            obj_class = 5
                        elif 'emergency' in name['category_name']:
                            obj_class = 6
                        elif 'construction' in name['category_name']:
                            obj_class = 7
                        elif 'movable_object' in name['category_name']:
                            obj_class = 8
                        elif 'bicycle_rack' in name['category_name']:
                            obj_class = 9

                        temp = [str(obj_class), str(x), str(y), str(w), str(h), '\n']
                        L = " "
                        L = L.join(temp)
                        f.writelines(L)
                f.close()
            n = open('nuscenes.names', "w+")
            n.write('pedestrian \n')
            n.write('bicycle \n')
            n.write('motorcycle \n')
            n.write('car \n')
            n.write('bus \n')
            n.write('truck \n')
            n.write('emergency \n')
            n.write('construction \n')
            n.write('movable_object \n')
            n.write('bicycle_rack \n')
            n.close()

    write_training_data_path_synced_with_labels(label_path)



if __name__ == '__main__':
    args = parse_arguments()

    if args.data_type == 'yolo':
        data = yolo_parser(args.label_dir, args.save_dir)

    elif args.data_type == 'bdd':
        data = bdd_parser(args.label_dir)

    elif args.data_type == 'nuscenes':
        data = bdd_parser(args.label_dir, args.save_dir, args.image_dir)

    else:
        print(40 * '-')
        print('{} data is not included in this parser!'.format(args.data_type))
