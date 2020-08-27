
#all: download-json download-img tweak-json

download-json:
	# mkdir -p json/raw
	# download json files from google drive
	# json/raw/{train|validation|test}.json

download-img:
	mkdir -p img/train img/validation img/test
	python3 download_img.py > log.txt

tweak-json:
	mkdir -p json/tweak
	python3 tweak_json.py

clean: clean-json clean-img

clean-json:
	rm -rf json

clean-img:
	rm -rf img log.txt
