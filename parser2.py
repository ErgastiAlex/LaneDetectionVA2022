
import os, getopt
import json
import sys

from numpy import full

class Point:
    def __init__(self,point_list) -> None:
        self.y=int(point_list["y"])
        self.x=int(point_list["x"])



line_min_height=20
line_min_marker=2


lane_list=["l1","l0","r0","r1"]

def main(argv):
    input_path,output_path,full_line=parse_argv(argv)

    for filename in os. listdir(input_path):
        filtered_lanes=get_filtered_lanes(os.path.join(input_path,filename))
        lines=get_lines(filtered_lanes,full_line)

        file_content=""
        for lane in lane_list:
            file_content+=" ".join(lines[lane])+"\n"

        with open(os.path.join(output_path,filename[:-5]),"w") as f:
            f.write(file_content)



def parse_argv(argv):
    input_path=""
    output_path=""
    full_line=False
    try:
      opts, args = getopt.getopt(argv,"hi:o:f",["inputdir=","outputdir=","full="])
    except getopt.GetoptError:
      print('test.py -i <inputdir> -o <outputpath>')
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
        print('test.py -i <inputdir> -o <outputdir>')
        sys.exit()
      elif opt in ("-i", "--inputdir"):
        input_path = arg
      elif opt in ("-o", "--outputdir"):
        output_path = arg
      elif opt in ("-f", "--full"):
        full_line=True
    
    if(input_path=="" or output_path==""):
        print('test.py -i <inputdir> -o <outputdir> -f')
        sys.exit()

    return input_path,output_path,full_line



def get_filtered_lanes(filename):
    with open(filename,"r") as f:
        json_file=json.load(f)
    
    filtered_lanes=remove_lines_to_short(json_file["lanes"])
    filtered_lanes=remove_lines_with_too_few_markers(filtered_lanes)
    filtered_lanes=rename_lanes(filtered_lanes)
    return filtered_lanes

def remove_lines_to_short(lanes):
    filtered_lane=[]

    for lane in lanes:
        #from top to bottom
        min_y=min(int(marker["pixel_start"]["y"]) for marker in lane["markers"])
        max_y=max(int(marker["pixel_start"]["y"]) for marker in lane["markers"])

        if(max_y-min_y<line_min_height):
            continue
        else:
            filtered_lane.append(lane)

    return filtered_lane

def remove_lines_with_too_few_markers(lanes):
    filtered_lane=[]

    for lane in lanes:
        if len(lane["markers"])<line_min_marker:
            continue
        else:
            filtered_lane.append(lane)
    return filtered_lane

def rename_lanes(filtered_lanes):
    left_lane=[]
    right_lane=[]
    for lane in filtered_lanes:
        if(lane["lane_id"][0]=="l"):
            left_lane.append(lane)
        else:
            right_lane.append(lane)
    left_lane.sort(key=lambda x: x["lane_id"])
    right_lane.sort(key=lambda x: x["lane_id"])

    left_counter=0
    for lane in left_lane:
        lane["lane_id"]="l"+str(left_counter)
        left_counter+=1

    right_counter=0
    for lane in right_lane:
        lane["lane_id"]="r"+str(right_counter)
        right_counter+=1
    left_lane.extend(right_lane)
    return left_lane
    

def get_lines(lanes,full_line):
    lines=dict()
    
    #basic initilization
    for lane in lane_list:
        lines[lane]=[str(-1) for _ in range(417)]


    for lane in lanes:
        if(lane["lane_id"] not in lane_list):
            continue

        line=calculate_line(lane["markers"],full_line)

        lines[lane["lane_id"]]=[str(l) for l in line]

    return lines

def calculate_line(markers,full_line):
    line=[-1 for i in range(717)]

    # Closest to the farttest point
    if(full_line):
        markers.sort(key=lambda x: x["pixel_start"]["y"],reverse=True)

    markers=filter_markers(markers)

    for i in range(0,len(markers)):
        m=markers[i]

        start = Point(m["pixel_start"])
        end   = Point(m["pixel_end"])

        if(full_line and i==0):
            line_from_start_to_end(line,start,end,start_y=717)
        else:
            line_from_start_to_end(line,start,end)

        


        #full line and not last marker
        if(full_line and i!=len(markers)-1):
            start=end
            end=Point(markers[i+1]["pixel_start"])

            line_from_start_to_end(line,start,end)

    #We keep only the element from 300 to 717
    return line[300:]

def filter_markers(markers):
    filtered_markers=[]

    for m in markers:
        start = Point(m["pixel_start"])
        end   = Point(m["pixel_end"])
        #We want to calculate the X given the Y
        if(start.y==end.y): #if the line is vertical
            continue
        if(start.x==end.x): #if the line is horizontal
            continue
        if(start.y-end.y<=2): # if the line is too short
            continue

        filtered_markers.append(m)

    return filtered_markers

def line_from_start_to_end(line,start,end,start_y=-1):  

    if(end.y-start.y==0):
        return
    slope=(end.x-start.x)/float(end.y-start.y)

    if(start_y==-1):
        start_y=start.y

    for y in range(end.y,start_y):
        x=start.x+slope*(y-start.y)
        if(x>=0 and x<1276):
            line[y]=x


if __name__ == "__main__":
    main(sys.argv[1:])