import cv2
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error


img1 = cv2.imread('image.png')
img2 = cv2.imread('output/_compressed_image.png_.jpg')

MSE = mean_squared_error(img1, img2)
PSNR = peak_signal_noise_ratio(img1, img2)
SSIM = structural_similarity(img1, img2, multichannel=True)


print('MSE: ', MSE)
print('PSNR: ', PSNR)
print('SSIM: ', SSIM)

# test
import webp
