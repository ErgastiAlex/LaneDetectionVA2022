import argparse
import json
import math
import os

from unsupervised_llamas.label_scripts import dataset_constants
from unsupervised_llamas.common import helper_scripts
from unsupervised_llamas.label_scripts import spline_creator

path_labelcpp="./test/labelcpp"
path_labeljson="./test/labeljson"

def calculate_lines(split):
    labels = helper_scripts.get_labels(split=split)

    jsonfile=dict()

    if(os.path.exists(os.path.join(path_labelcpp,split))==False):
        os.mkdir(os.path.join(path_labelcpp,split))

    for label in labels:
        spline_labels = spline_creator.get_horizontal_values_for_four_lanes(label)
        assert len(spline_labels) == 4, "Incorrect number of lanes"
        key = helper_scripts.get_label_base(label)
        key_without_ext=key[:-5]

        dir,filename=key_without_ext.split("/")

        jsonfile[filename]=dict()

        cppfile=""
        
        for lane, lane_key in zip(spline_labels, ["l1", "l0", "r0", "r1"]):
            cppfile += " ".join([str(l) for l in lane[300:]]) +"\n"
            jsonfile[filename][lane_key]=lane[300:]

        if(os.path.exists(os.path.join(path_labelcpp,split,dir))==False):
            os.mkdir(os.path.join(path_labelcpp,split,dir))

        with open(os.path.join(path_labelcpp,split,key_without_ext),"w") as f:
            f.write(cppfile)
    
    with open(os.path.join(path_labeljson,split+".json"),"w") as f:
        f.write(json.dumps(jsonfile))  
    

calculate_lines("valid")