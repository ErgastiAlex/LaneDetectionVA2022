
VIDEOPATH?=imageswithline/
default_target: nothing

nothing:
	@echo "This is the default target.  No target was provided."

parser_full_line:
	python3 parser.py -i labels/valid/images-2014-12-22-12-35-10_mapping_280S_ramps/ -o test2 -f

parser:
	python3 parser.py -i labels/valid/images-2014-12-22-12-35-10_mapping_280S_ramps/ -o test2

video: 
	cat ${VIDEOPATH}*.png | ffmpeg -f image2pipe -i - output.mkv
	echo "Video created"