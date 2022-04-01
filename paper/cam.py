import argparse
import os

import cv2
import numpy as np
import torch
from torchvision import models


from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from compression.params import HyperParams
hyper = HyperParams(verbose=False)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        # default='./examples/both.png',
        default='dataset/Kodak/image.png',
        help='Input image path')
    parser.add_argument('-single', type=int, default=0) #1
    parser.add_argument('-dataset', type=str, default='kodak')

    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam++',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}



    ### trained model by ourself
    if hyper.finetune:
        model = models.vgg16(pretrained=True)  ##
        # model_conv.classifier._modules['6'] = nn.Linear(4096, 2)
        ckpt = torch.load('./model/vgg16_.ckpt' )
        model.load_state_dict(ckpt['net_state_dict'])

    model = models.vgg16(pretrained=True)  ##
    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    target_layers = [model.features[-1]]


    if args.single:
        rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
        # rgb_img = cv2.resize(rgb_img, (hyper.image_h,hyper.image_w))  # msroi
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])


        # We have to specify the target we want to generate
        # the Class Activation Maps for.
        # If targets is None, the highest scoring category (for every member in the batch) will be used.
        # You can target specific categories by
        # targets = [e.g ClassifierOutputTarget(281)]
        targets = None

        # Using the with statement ensures the context is freed, and you can
        # recreate different CAM objects in a loop.
        cam_algorithm = methods[args.method]
        with cam_algorithm(model=model,
                           target_layers=target_layers,
                           use_cuda=args.use_cuda) as cam:

            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 32
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            # roi_image = show_cam_on_image(rgb_img, roi_map, use_rgb=True)
            # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            # roi_image = cv2.cvtColor(roi_image, cv2.COLOR_RGB2BGR)

        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
        gb = gb_model(input_tensor, target_category=None)

        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)
        if hyper.mscam and hyper.mscam_softmax:
            cv2.imwrite(f'compression/input/{args.method}_{args.image_path[-9:-4]}_ms{str(hyper.top_k)}soft_cam.jpg', cam_image)
        elif hyper.mscam:
            cv2.imwrite(f'compression/input/{args.method}_{args.image_path[-9:-4]}_ms{str(hyper.top_k)}_cam.jpg',
                        cam_image)
        else:
            cv2.imwrite(f'compression/input/{args.method}_{args.image_path[-9:-4]}_cam.jpg',
                        cam_image)
        # cv2.imwrite(f'compression/input/{args.method}_roi.jpg', roi_image)
        # cv2.imwrite(f'{args.method}_gb.jpg', gb)
        # cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb)
        # save cam image
        import skimage.io
        # roi_map = np.uint8(255 * roi_map)
        # get cam_map, like msroi_map
        # cam_map = 255 - np.uint8(255 * grayscale_cam)
        cam_map = np.uint8(255 * grayscale_cam)
        if hyper.mscam and hyper.mscam_softmax:
            skimage.io.imsave(f'compression/input/{args.method}_{args.image_path[-9:-4]}_ms{str(hyper.top_k)}soft_cam_map.jpg', cam_map)
        elif hyper.mscam:
            skimage.io.imsave(
                f'compression/input/{args.method}_{args.image_path[-9:-4]}_ms{str(hyper.top_k)}_cam_map.jpg', cam_map)
        else:
            skimage.io.imsave(
                f'compression/input/{args.method}_{args.image_path[-9:-4]}_cam_map.jpg', cam_map)
    else:
        output_pre = 'compression/input/'+str(args.dataset)+'/'
        if not os.path.exists(output_pre):
            os.makedirs(output_pre)
        from glob import glob
        if args.dataset == 'kodak':
            image_path = 'dataset/kodak/*.png'
        else:
            assert Exception("Wrong dataset choosen")
        for image_file in glob(image_path):
            print(image_file)
            args.image_path = image_file
            if args.dataset == 'kodak':
                image_num = args.image_path.split('/')[-1].split('.')[0]
            rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
            # rgb_img = cv2.resize(rgb_img, (hyper.image_h,hyper.image_w))  # msroi
            rgb_img = np.float32(rgb_img) / 255
            input_tensor = preprocess_image(rgb_img,
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

            # We have to specify the target we want to generate
            # the Class Activation Maps for.
            # If targets is None, the highest scoring category (for every member in the batch) will be used.
            # You can target specific categories by
            # targets = [e.g ClassifierOutputTarget(281)]
            targets = None

            # Using the with statement ensures the context is freed, and you can
            # recreate different CAM objects in a loop.
            cam_algorithm = methods[args.method]
            with cam_algorithm(model=model,
                               target_layers=target_layers,
                               use_cuda=args.use_cuda) as cam:

                # AblationCAM and ScoreCAM have batched implementations.
                # You can override the internal batch size for faster computation.
                cam.batch_size = 32
                grayscale_cam = cam(input_tensor=input_tensor,
                                    targets=targets,
                                    aug_smooth=args.aug_smooth,
                                    eigen_smooth=args.eigen_smooth)

                # Here grayscale_cam has only one image in the batch
                grayscale_cam = grayscale_cam[0, :]

                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                # roi_image = show_cam_on_image(rgb_img, roi_map, use_rgb=True)
                # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                # roi_image = cv2.cvtColor(roi_image, cv2.COLOR_RGB2BGR)

            gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
            gb = gb_model(input_tensor, target_category=None)

            cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
            cam_gb = deprocess_image(cam_mask * gb)
            gb = deprocess_image(gb)
            if hyper.mscam and hyper.mscam_softmax:
                cv2.imwrite(
                    f'{output_pre}{args.method}_{image_num}_ms{str(hyper.top_k)}soft_cam.jpg',
                    cam_image)
            elif hyper.mscam:
                cv2.imwrite(
                    f'{output_pre}{args.method}_{image_num}_ms{str(hyper.top_k)}_cam.jpg',
                    cam_image)
            else:
                cv2.imwrite(f'{output_pre}{args.method}_{image_num}_cam.jpg',
                            cam_image)
            # cv2.imwrite(f'compression/input/{args.method}_roi.jpg', roi_image)
            # cv2.imwrite(f'{args.method}_gb.jpg', gb)
            # cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb)
            # save cam image
            import skimage.io

            # roi_map = np.uint8(255 * roi_map)
            # get cam_map, like msroi_map
            # cam_map = 255 - np.uint8(255 * grayscale_cam)
            cam_map = np.uint8(255 * grayscale_cam)
            if hyper.mscam and hyper.mscam_softmax:
                skimage.io.imsave(
                    f'{output_pre}{args.method}_{image_num}_ms{str(hyper.top_k)}soft_cam_map.jpg',
                    cam_map)
            elif hyper.mscam:
                skimage.io.imsave(
                    f'{output_pre}{args.method}_{image_num}_ms{str(hyper.top_k)}_cam_map.jpg',
                    cam_map)
            else:
                skimage.io.imsave(
                    f'{output_pre}{args.method}_{image_num}_cam_map.jpg', cam_map)

    # skimage.io.imsave(f'compression/input/{args.method}_roi_map.jpg', roi_map)
    # skimage.io.imsave('output/color_cam_map1.jpg', cam_image)
    import matplotlib.pyplot as plt
    from PIL import Image
    # cam_map = Image.fromarray(cam_map)
    # plt.imshow(roi_map, cmap=plt.cm.jet, interpolation='nearest')

