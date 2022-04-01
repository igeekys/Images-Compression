import numpy as np
import torch
import ttach as tta
from typing import Callable, List, Tuple

from compression.util import normalize
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from compression.params import HyperParams
hyper = HyperParams(verbose=False)

class BaseCAM:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True) -> None:
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

        # MSROI: get W of 'GAP'
        # self.fc_weights = self.model.fc.weight

    # MSROI: get classmap. this is for CAM, need train the model again with depthwise_conv2d_native
    # def get_classmap(self,class_,conv_last,input_tensor):
    #
    #     class_w = self.fc_weights[class_,:]  #Size([2048])
    #     target_size = self.get_target_width_height(input_tensor)
    #     conv_last_ = torch.nn.functional.interpolate(conv_last, target_size, mode='bilinear',
    #                                         align_corners=False)
    #     h, w, c = conv_last_.shape[2], conv_last_.shape[3], conv_last_.shape[1]
    #     conv_last_ = conv_last_.reshape(-1,c,w*h)
    #     classmap = torch.matmul(class_w,conv_last_).reshape(-1,h,w)
    #     return torch.transpose(classmap,1,2)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        target_layers: List[torch.nn.Module],
                        targets: List[torch.nn.Module],
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:
        raise Exception("Not Implemented")

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        ## get grad belong to one class
        weights = self.get_cam_weights(input_tensor,
                                       target_layer,
                                       targets,
                                       activations,
                                       grads)
        ##
        weighted_activations = weights[:, :, None, None] * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:

        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)
        outputs = self.activations_and_grads(input_tensor)
        ## msroi start:  this is for CAM, need train the model again with depthwise_conv2d_native
        # conv_last = self.activations_and_grads.activations[0]
        # # use argsort instead of argmax to get all the classes
        # class_predictions_all = outputs.argsort()
        # roi_map = None
        # for i in range(-1 * 1, 0):  #
        #     current_class = class_predictions_all[:,i]
        #     classmap_vals = self.get_classmap(current_class,conv_last,input_tensor)
        #     normalized_classmap = normalize(classmap_vals[0].cpu().data.numpy())  # [0]?
        #     if roi_map is None:
        #         roi_map = 1.2 * normalized_classmap
        #     else:
        #         # simple exponential ranking
        #         roi_map = (roi_map + normalized_classmap) / 2
        # roi_map = normalize(roi_map)
        ## msroi end:


        ## mscam :multi-class, this is for grad CAM, don't need train again.
        # imagenet1000 dataset is diff with celtech256 in msroi
        if hyper.mscam:
            class_predictions_all = np.argsort(outputs.cpu().data.numpy(), axis=-1)
            target_categories = [class_predictions_all[:,i] for i in range(-1 * hyper.top_k, 0)]  #5/256 == 20/1000
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            target_outputs = torch.cat([target(outputs) for target in targets])
            if hyper.mscam and hyper.mscam_softmax:
                # mscam softmax:
                t = torch.nn.Softmax(dim=0)(target_outputs)  # softmax
                loss = torch.matmul(torch.transpose(t, 0, 1), target_outputs).reshape(1)
            else:
                ## multi class add
                loss = sum(target_outputs.reshape(-1,1))
            ### loss = sum([target(output) for target, output in zip(targets, outputs)])  ## just one output
            loss.backward(retain_graph=True)
        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int]:
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> np.ndarray:
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            cam = np.maximum(cam, 0)
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result)

    def forward_augmentation_smoothing(self,
                                       input_tensor: torch.Tensor,
                                       targets: List[torch.nn.Module],
                                       eigen_smooth: bool = False) -> np.ndarray:
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                               targets,
                               eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor: torch.Tensor,
                 targets: List[torch.nn.Module] = None,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False) -> np.ndarray:

        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor,
                            targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True