#Copyright 2015 Yale University - Grablab
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import sys
import json
import urllib
import urllib.request as urllib2

output_directory = "./object_datasets/ycb"

# You can either set this to "all" or a list of the objects that you'd like to
# download.
#objects_to_download = "all"
#objects_to_download = ["002_master_chef_can", "003_cracker_box"]
# objects_to_download = ["035_power_drill", 
#                        "036_wood_block", 
#                        "037_scissors", 
#                        "038_padlock", 
#                        "040_large_marker", 
#                        # "041_small_marker", 
#                        "042_adjustable_wrench",
#                        # "043_phillips_screwdriver",
#                        "044_flat_screwdriver",
#                        "048_hammer",
#                        "049_small_clamp",
#                        "050_medium_clamp",
#                        "052_extra_large_clamp",
#                     ]

objects_to_download = ["035_power_drill", 
                       "036_wood_block", 
                       "054_softball",
                       "006_mustard_bottle",
                       "001_chips_can",
                       "010_potted_meat_can",
                       "002_master_chef_can",
                       "004_sugar_box",
                       "065-h_cups",
                       "024_bowl",
                       "025_mug"
                    ]

# You can edit this list to only download certain kinds of files.
# 'berkeley_rgbd' contains all of the depth maps and images from the Carmines.
# 'berkeley_rgb_highres' contains all of the high-res images from the Canon cameras.
# 'berkeley_processed' contains all of the segmented point clouds and textured meshes.
# 'google_16k' contains google meshes with 16k vertices.
# 'google_64k' contains google meshes with 64k vertices.
# 'google_512k' contains google meshes with 512k vertices.
# See the website for more details.
#files_to_download = ["berkeley_rgbd", "berkeley_rgb_highres", "berkeley_processed", "google_16k", "google_64k", "google_512k"]
files_to_download = ["google_16k"]

# Extract all files from the downloaded .tgz, and remove .tgz files.
# If false, will just download all .tgz files to output_directory
extract = True

base_url = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"
objects_url = base_url + "objects.json"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def fetch_objects(url):
    response = urllib2.urlopen(url)
    html = response.read()
    objects = json.loads(html)
    return objects["objects"]

def download_file(url, filename):
    u = urllib2.urlopen(url)
    f = open(filename, 'wb')
    meta = u.info()
    file_size = int(meta.get("Content-Length")[0])
    print("Downloading: %s (%s MB)" % (filename, file_size/1000000.0))

    file_size_dl = 0
    block_sz = 65536
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl/1000000.0, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print(status)
    f.close()

def tgz_url(object, type):
    if type in ["berkeley_rgbd", "berkeley_rgb_highres"]:
        return base_url + "berkeley/{object}/{object}_{type}.tgz".format(object=object,type=type)
    elif type in ["berkeley_processed"]:
        return base_url + "berkeley/{object}/{object}_berkeley_meshes.tgz".format(object=object,type=type)
    else:
        return base_url + "google/{object}_{type}.tgz".format(object=object,type=type)

def extract_tgz(filename, dir):
    tar_command = "tar -xzf {filename} -C {dir}".format(filename=filename,dir=dir)
    os.system(tar_command)
    os.remove(filename)

def check_url(url):
    try:
        request = urllib2.Request(url)
        request.get_method = lambda : 'HEAD'
        response = urllib2.urlopen(request)
        return True
    except Exception as e:
        return False


if __name__ == "__main__":

    objects = objects_to_download#fetch_objects(objects_url)

    for object in objects:
        if objects_to_download == "all" or object in objects_to_download:
            for file_type in files_to_download:
                url = tgz_url(object, file_type)
                if not check_url(url):
                    continue
                filename = "{path}/{object}_{file_type}.tgz".format(path=output_directory,
                                                                    object=object,
                                                                    file_type=file_type)
                download_file(url, filename)
                if extract:
                    extract_tgz(filename, output_directory)
    # find obj files
    obj_files = []
    obj_dir = './object_datasets/ycb'
    for root, dirs, files in os.walk(obj_dir):
        if 'poisson' in root:
            continue
        for file in files:
            if file.endswith(".obj"):
                obj_files.append(os.path.join(root, file))
    # move obj files to individual directories
    os.makedirs('./object_datasets/ycb/meshes', exist_ok=True)
    f = open('./object_datasets/ycb/meshes/obj_files.txt', 'w')
    for i, obj_file in enumerate(obj_files):
        os.rename(obj_file, f'./object_datasets/ycb/meshes/{i}.obj')
        f.write(f'Object id #{i}: {obj_file}\n')
    f.close()
