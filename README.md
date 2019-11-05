2D label parser for Yolov3 darknet
---
This repository contains the parser functions to convert COCO/YOLO, BDD, and nuscenes json format to txt format needed by darknet Yolov3.

2D bounding box format is taken as [xmin, xmax, ymin, ymax]. Moreover, the object classes from nuscenes dataset is reduced from 23 to 10 regarding our needs; "pedestrian, bicycle, motorcycle, car, bus, truck, emergency, construction, movable object, and bicycle_rack" are the objects of interest in this parser. You can eaily customize it according to your needs in the nuscenes_parser function.

Install conda environment
---
Python pachages needed to run the parser is exported to "environement.yaml". You can see the env name and dependencies by typing "more environment.yaml".

To create the conda environment, type the following  in the terminal:
"conda env create -f environment.yaml"
