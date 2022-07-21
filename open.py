import cv2
import numpy as np
import math
import sys
import os, getopt

line_color={"l0":[255,0,0],"l1":[0,255,0],"r0":[0,0,255],"r1":[255,255,0]} # l0 is blue, l1 is green, r0 is red, r1 is yellow
line_list=["l1","l0","r0","r1"]

def main(argv):
    filename,is_test2=get_args(argv)
    img=cv2.imread(f"/media/ergale/SSD/Universita/Parma - Scienze Informatiche/1INF/2SEM/Visione artificiale per il veicolo/LaneDetection/image/gray/images-2014-12-22-12-35-10_mapping_280S_ramps/{filename}_gray_rect.png",cv2.CV_8UC1)   
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    if is_test2:
        file=open(f"/media/ergale/SSD/Universita/Parma - Scienze Informatiche/1INF/2SEM/Visione artificiale per il veicolo/LaneDetection/test2/{filename}","r")
    else:
        file=open(f"/media/ergale/SSD/Universita/Parma - Scienze Informatiche/1INF/2SEM/Visione artificiale per il veicolo/LaneDetection/test/labelcpp/valid/images-2014-12-22-12-35-10_mapping_280S_ramps/{filename}","r")

    line_count=0
    for line in file.readlines():
        line_coord=[float(x) for x in line.split(" ")]
        base_y=0
        if len(line_coord)==417:
          base_y=300
        for y in range(len(line_coord)):

            if line_coord[y]!=-1:
                img[y+base_y,math.floor(line_coord[y])]=line_color[line_list[line_count]]

        line_count+=1

    # img3=np.zeros((img.shape[0],img.shape[1],3),dtype="uint8")
    # cv2.merge([img,img,img2],img3)

    cv2.imshow("img",img)

    cv2.waitKey(0)

def get_args(argv):
    filename=""
    is_test2=True

    try:
      opts, args = getopt.getopt(argv,"hf:t:",["filename=","test2="])
    except getopt.GetoptError:
      print('test.py -f <filename>')
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
        print('test.py -f <filename>')
        sys.exit()
      elif opt in ("-f", "--filename"):
        filename=arg
      elif opt in ("-t", "--test2"):
        if(arg=="1"):
            is_test2=True
        else:
            is_test2=False
    
    if(filename==""):
        print('test.py -f <filename>')
        sys.exit()

    return filename,is_test2


if __name__ == "__main__":
    main(sys.argv[1:])