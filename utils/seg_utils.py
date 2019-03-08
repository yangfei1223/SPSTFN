import numpy as np
import matplotlib.cm as cm
from scipy.misc import imread, imshow, imsave, toimage
import os


def make_overlay(image, gt_prob):

    mycm = cm.get_cmap('bwr')

    overimage = mycm(gt_prob, bytes=True)
    output = 0.4*overimage[:, :, 0:3] + 0.6*image

    return output


def fast_overlay(input_image, segmentation, color=[0, 255, 0, 127]):
    """
    Overlay input_image with a hard segmentation result for two classes.

    Store the result with the same name as segmentation_image, but with
    `-overlay`.

    Parameters
    ----------
    input_image : numpy.array
        An image of shape [width, height, 3].
    segmentation : numpy.array
        Segmentation of shape [width, height].
    color: color for forground class

    Returns
    -------
    numpy.array
        The image overlayed with the segmenation
    """
    color = np.array(color).reshape(1, 4)
    shape = input_image.shape
    segmentation = segmentation.reshape(shape[0], shape[1], 1)

    output = np.dot(segmentation, color)
    output = toimage(output, mode="RGBA")

    background = toimage(input_image)
    background.paste(output, box=None, mask=output)

    return np.array(background)


def overlayImageWithConfidence(in_image, conf, vis_channel=1, threshold=0.5):
    '''

    :param in_image:
    :param conf:
    :param vis_channel:
    :param threshold:
    '''
    if in_image.dtype == 'uint8':
        visImage = in_image.copy().astype('f4') / 255
    else:
        visImage = in_image.copy()

    channelPart = visImage[:, :, vis_channel] * (conf > threshold) - conf
    channelPart[channelPart < 0] = 0
    visImage[:, :, vis_channel] = visImage[:, :, vis_channel] * (conf <= threshold) + (
                                                                                      conf > threshold) * conf + channelPart
    return visImage


rgb_dir = '/home/yangfei/Desktop/experiment_on_val/val_rgb_bev'
prob_dir = '/home/yangfei/Desktop/experiment_on_val/val_bev'
save_dir = '/home/yangfei/Desktop/experiment_on_val/val_bev_vis'
filelist = os.listdir(rgb_dir)

for filename in filelist:
    print filename
    rgb = imread(os.path.join(rgb_dir, filename), mode='RGB')
    prob = imread(os.path.join(prob_dir, filename), mode='L')
    im_rb = make_overlay(rgb, prob)
    imsave(os.path.join(save_dir, filename.split('.')[0] + '_rb.png'), im_rb)
    im_green = fast_overlay(rgb, prob > 128)
    imsave(os.path.join(save_dir, filename.split('.')[0] + '_green.png'), im_green)






