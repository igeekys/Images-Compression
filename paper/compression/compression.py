# from __future__ import division
from email.mime import image
import sys
from PIL import Image
import os
import numpy as np
from util import load_image, array2PIL
import argparse
from scipy.stats import percentileofscore
from glob import glob


def make_quality_compression(original,sal):
    if args.print_metrics:
        print(args.image, end=' ')
    # if the size of the map is not the same original image, then blow it
    uncompressed_size = original.size
    if original.size != sal.size:
        sal = sal.resize(original.size)

    sal_arr = np.array(sal)
    img_qualities = []
    quality_steps = [i*10 for i in range(1,11)]

    # this temp directory will be deleted, do not use this to store your files
    os.makedirs('temp_xxx_yyy')
    for q in quality_steps:
        name = 'temp_xxx_yyy/temp_' + str(q) + '.jpg'
        if args.use_convert:
            os.system('convert -colorspace sRGB -filter Lanczos -interlace Plane -type truecolor -quality ' + str(q) + ' ' + args.image + ' ' + name)
        else:
            original.save(name, quality=q)
        img_qualities.append(np.array(Image.open(name)))
        os.remove(name)
    os.rmdir('temp_xxx_yyy')
                   
    k = img_qualities[-1][:] #== original make sure it is a copy and not reference
    shape = k.shape 
    k.flags.writeable = True
    mx, mn = np.max(sal_arr), np.mean(sal_arr)
    sal_flatten = sal_arr.flatten()

    # 分位数就是用概率作为依据将一批数据分开的那个点
    q_2,q_3,q_5,q_6,q_9 = map(lambda x: np.percentile(sal_arr, x), [20,30,50,60,90])

    q_a = [np.percentile(sal_arr, j) for j in quality_steps]
    low, med, high = 1, 5, 9

    for i in range(shape[0]):
        for j in range(shape[1]):
            for l in range(shape[2]):
                ss = sal_arr[i,j]

                if args.model == 1:
                    # model -1 
                    # hard-coded model
                    if ss > mn: qq = 9
                    else: qq = 6

                elif args.model == 2:
                    # model -2 
                    # linearly scaled technique
                    qq = (ss * 10 // mx) -1  + 3
                
                elif args.model == 3:  ##default
                    # model -3 
                    # percentile based technique
                    # qq = int(percentileofscore(sal_flatten, ss)/10)
                    for index, q_i in enumerate(q_a):
                        if ss < q_i: 
                            qq = index + 1
                            break

                elif args.model == 4:
                    # model -4 
                    # discrete percentile based technique
                    # if   ss < q_2: qq = 4 
                    if ss < q_2: qq = 4 
                    elif ss < q_6: qq = 6 
                    elif ss < q_9: qq = 8 
                    else: qq = 9

                elif args.model == 5:
                    # model -5
                    # two way percentile
                    if ss <  q_5: qq = 2
                    else: qq = 8

                elif args.model == 6:
                    # model -6
                    # two way percentile - higher coverage
                    if ss <  q_5: qq = 7
                    else: qq = 9

                elif args.model == 7:
                    # model -6
                    # two way percentile - higher coverage
                    if ss <  q_5: qq = 6
                    else: qq = 10
                    
                else:
                    raise Exception("unknown model number")

                if qq < low : qq = low
                if qq > high: qq = high 
                k[i,j,l] = img_qualities[qq][i,j,l]

    # save the original file at the given quality level
    compressed = args.output_directory + '/' + 'original_' + args.image.split('/')[-1] + '_' + str(args.jpeg_compression) + '_' + args.map.split('/')[-1]
    original.save(compressed, quality=args.jpeg_compression)  ##? save can compress? yes
    original_size = os.path.getsize(compressed)

    ## bpg begin
    # ? aim for what? can't usr roi_map or cam_map
    # compressed_1 = args.output_directory + '/' + '_original_' + args.image.split('/')[-1] + '_' + str(
    #     args.jpeg_compression) + '_tmp' + '.jpg'
    # compressed_2 = args.output_directory + '/' + '_original_' + args.image.split('/')[-1] + '_' + str(
    #     args.jpeg_compression) + '_' + '.bpg'
    # original.save(compressed_1, quality=100)
    # os.system('bpgenc -q ' + str(args.jpeg_compression) + ' ' + compressed_1 + ' ' + compressed_2)
    ## bpg end


    # os.system('convert ' + args.image + " " + args.output_directory + '/temp.png')
    # uncompressed_size = os.path.getsize(args.output_directory + '/temp.png')
    # os.remove(args.output_directory + '/temp.png')

    out_img = array2PIL(k)
    out_img = out_img.convert('RGB')  # 新加一行 for [python] OSError: cannot write mode RGBA as JPEG

    if args.find_best:
        out_name = args.output_directory + '/' + 'compressed_' + args.image.split('/')[-1] + '_' + str(args.jpeg_compression) + '_'  + args.map.split('/')[-1]
        for qual in range(90,20,-1):
            out_img.save(out_name, quality=qual)
            current_size = os.path.getsize(out_name)
            if current_size<= original_size*(1 + args.threshold_pct/100.0): 
                if args.print_metrics:
                    print(args.model, uncompressed_size, original_size, current_size, args.jpeg_compression, qual,
                          ' | ', end=' ')
                break
        else:
            if args.print_metrics:
                print(args.model, uncompressed_size, original_size, current_size, args.jpeg_compression, qual, ' | ', end=' ')
            pass
    else:
        final_quality = [100, 85, 65, 45]
        for fq in final_quality:
            out_name = args.output_directory + '/' + args.modifier + args.image.split('/')[-1] + '_' + str(fq)  + args.map[6:-12]+ '.jpg'
            out_img.save(out_name, quality=fq)

    return compressed, out_name


def get_metrics(original, compressed, out_name, size):
    
    metrics = "PSNR SSIM MSSSIM VIFP PSNRHVS PSNRHVSM".lower().split(' ')
    # TODO add -y (overwrite files) 
    # first convert all the three images to yuv format
    size_x = size[0] - size[0]%16 # this is to make sure we can get MS-SSIM 
    size_y = size[1] - size[1]%16 # metrics from VQMT, which requires divisible by 16

    for x in [original, compressed, out_name]:
        yuv_convert_command = "ffmpeg -hide_banner -loglevel panic -y -i " + x +" -s " + str(size_x) + "x" + str(size_y) + " -pix_fmt yuv420p " + x +".yuv"
        if os.system(yuv_convert_command) != 0:
            raise Exception("FFMPEG was not found")
        # print command
    for img_com in [compressed, out_name]:
        ## need modify
        command_metrics = "/mnt/md0p1/GPH/AAAA/VQMT-master/build/bin/Release/vqmt " + \
                          original+".yuv " + \
                          img_com+".yuv " + \
                          str(size_x) + " " + \
                          str(size_y) + " " + \
                          "1 1 out PSNR SSIM MSSSIM VIFP PSNRHVS PSNRHVSM"
        if os.system(command_metrics) != 0:
            raise Exception("VQMT was not found, please install it from https://github.com/Rolinh/VQMT")
        for m in metrics:
            f = open('out_' + m + '.csv').read().splitlines()[1].split(',')[1]
            print(f, end=' ')
        print(' | ', end=' ')
    print('')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-image'            , type=str , default= '../dataset/Kodak/image.png')
    parser.add_argument('-map'              , type=str , default= 'input/ROI/gradcampp.jpg')
    parser.add_argument('-output_directory' , type=str , default= 'output/res')
    parser.add_argument('-modifier'         , type=str , default= '')
    parser.add_argument('-find_best'        , type=int , default=1)
    # change the threshold % to 1, if doing metric comparison against standard JPEG. 
    # Images will have limited discernibility but fairer comparison against standard.
    parser.add_argument('-threshold_pct'    , type=int , default=20) # 20
    # if you have Imagemagick installed, use convert it is faster
    parser.add_argument('-use_convert'      , type=int , default=0) # 0
    # try at multiple values. 50 is standard for our paper
    parser.add_argument('-jpeg_compression' , type=int , default=50)
    # there are various models from 1 to 6 on how best to mix different JPEG Qualities
    # ??
    parser.add_argument('-model'            , type=int , default=7) # default 3
    parser.add_argument('-single'           , type=int , default=0)
    parser.add_argument('-dataset'          , type=str , default='kodak')
    # printing metrics requires https://github.com/Rolinh/VQMT
    parser.add_argument('-print_metrics'    , type=int , default=0)
    args = parser.parse_args()


    # make the output directory to store the Q level images, 
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # if args.print_metrics:
    #     from get_metrics import get_metrics

    if args.single:
        original = Image.open(args.image)
        sal = Image.open(args.map)
        a,b = make_quality_compression(original,sal)  # a b are paths
        if args.print_metrics:
            get_metrics(args.image, a, b, original.size)

    else:
        if args.dataset == 'kodak':
            image_path = '../dataset/Kodak/*.png'
        elif args.dataset == 'large':
            image_path = 'images_directory/output_large/ori_*.png'
        else:
            assert Exception("Wrong dataset choosen")

        for image_file in glob(image_path):
            if args.dataset == 'large':
                map_file = 'images_directory/output_large/map' + image_file.split('/')[-1][3:-4] 
            elif args.dataset == 'kodak':
                map_file = 'input/kodak/gradcam_kodak/' + image_file.split('/')[-1].split('\\')[-1].split('.')[0] + '_cam_map.jpg'
            args.image = image_file
            args.map   = map_file
            original = Image.open(args.image)
            sal = Image.open(args.map)
            a,b = make_quality_compression(original,sal)
            if args.print_metrics:
                get_metrics(args.image, a, b, original.size)
