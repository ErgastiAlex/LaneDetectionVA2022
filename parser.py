
import os, getopt
import json
import sys

lane_set=("r1","r0","l1","l0")

def get_paths(argv):
    input_path=""
    output_filename=""

    try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","outputfilename="])
    except getopt.GetoptError:
      print('test.py -i <inputdir> -o <outputpath>')
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
        print('test.py -i <inputdir> -o <outputpath>')
        sys.exit()
      elif opt in ("-i", "--ifile"):
        input_path = arg
      elif opt in ("-o", "--outputpath"):
        output_filename = arg
    
    if(input_path=="" or output_filename==""):
        print('test.py -i <inputdir> -o <outputpath>')
        sys.exit()

    return input_path,output_filename

def parse_file(filename):
    r0=[]
    r1=[]
    l0=[]
    l1=[]

    with open(filename,"r") as f:
        json_file=json.load(f)
    
    lanes=json_file["lanes"]

    for obj_lane in lanes:
        if(obj_lane["lane_id"] not in lane_set):
            continue

        markers=obj_lane["markers"]

        for m in markers:
            x_start=m["pixel_start"]["x"]
            y_start=m["pixel_start"]["y"]
            x_end=m["pixel_end"]["x"]
            y_end=m["pixel_end"]["y"]

            if obj_lane["lane_id"]=="l0":
                l0.append([x_start,y_start])
                l0.append([x_end,y_end])
            elif obj_lane["lane_id"]=="l1":
                l1.append([x_start,y_start])
                l1.append([x_end,y_end])
            elif obj_lane["lane_id"]=="r0":
                r0.append([x_start,y_start])
                r0.append([x_end,y_end])
            elif obj_lane["lane_id"]=="r1":
                r1.append([x_start,y_start])
                r1.append([x_end,y_end])

    return l1,l0,r0,r1

def main(argv):
    input_path,output_filename=get_paths(argv)

    jsonfile=dict()
    for filename in os. listdir(input_path):
        l1,l0,r0,r1=parse_file(os.path.join(input_path,filename))
        
        jsonfile[filename]={"l1":l1,"l0":l0,"r0":r0,"r1":r1}
    
    with open(output_filename,"w") as f:
        f.write(json.dumps(jsonfile))

if __name__ == "__main__":
    main(sys.argv[1:])