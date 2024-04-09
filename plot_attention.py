import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np

dtype_range = {bool: (False, True),
               np.bool_: (False, True),
               np.bool8: (False, True),
               float: (-1, 1),
               np.float_: (-1, 1),
               np.float16: (-1, 1),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}

def peak_signal_noise_ratio(image_true, image_test, *, data_range=None):
    """
    Compute the peak signal to noise ratio (PSNR) for an image.

    Parameters
    ----------
    image_true : ndarray
        Ground-truth image, same shape as im_test.
    image_test : ndarray
        Test image.
    data_range : int, optional
        The dataa range of the input image (distance between minimum and
        maximum possible values).  By default, this is estimated from the image
        dataa-type.

    Returns
    -------
    psnr : float
        The PSNR metric.

    Notes
    -----
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_psnr`` to
        ``skimage.metrics.peak_signal_noise_ratio``.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    """
    assert image_true.shape == image_test.shape,"different shape between image_true and image_test"

    if data_range is None:
        if image_true.dtype != image_test.dtype:
            print("warning: Inputs have mismatched dtype.  Setting data_range based on "
                 "image_true.")
        dmin, dmax = dtype_range[image_true.dtype.type]
        true_min, true_max = np.min(image_true), np.max(image_true)
        if true_max > dmax or true_min < dmin:
            raise ValueError(
                "image_true has intensity values outside the range expected "
                "for its dataa type. Please manually specify the data_range.")
        if true_min >= 0:
            # most common case (255 for uint8, 1 for float)
            data_range = dmax
        else:
            data_range = dmax - dmin

    err = np.mean((image_true - image_test) ** 2)
    return 10 * np.log10((data_range ** 2) / err)


# img1 = r'/media/police1/buhaochi/cv/denoise_dataset/expriment/FFD/CBSD68/16077.png'
# img2 = r'/media/police1/buhaochi/cv/denoise_dataset/expriment/Deam/CBSD68/16077.png'
# img3 = r'/media/police1/buhaochi/cv/denoise_dataset/expriment/NB/CBSD68/16077.png'
# img4 = r'/media/police1/buhaochi/cv/denoise_dataset/expriment/TC/CBSD68/16077.png'
# img5 = r'/media/police1/buhaochi/cv/denoise_dataset/expriment/gt/CBSD68/16077.png'
# img6 = r'/media/police1/buhaochi/cv/denoise_dataset/expriment/noised_50/CBSD68/16077.png'

img1 = r'/media/police1/buhaochi/cv/denoise_dataset/enlarge/FFD/CBSD68/271035.png'
img2 = r'/media/police1/buhaochi/cv/denoise_dataset/enlarge/Deam/CBSD68/271035.png'
img3 = r'/media/police1/buhaochi/cv/denoise_dataset/enlarge/NB/CBSD68/271035.png'
img4 = r'/media/police1/buhaochi/cv/denoise_dataset/enlarge/TC/CBSD68/271035.png'
img5 = r'/media/police1/buhaochi/cv/denoise_dataset/enlarge/gt/CBSD68/271035.png'
img6 = r'/media/police1/buhaochi/cv/denoise_dataset/enlarge/noised_50/CBSD68/271035.png'

img1 = np.array(Image.open(img1))/255.0
img2 = np.array(Image.open(img2))/255.0
img3 = np.array(Image.open(img3))/255.0
img4 = np.array(Image.open(img4))/255.0
img5 = np.array(Image.open(img5))/255.0
img6 = np.array(Image.open(img6))/255.0



ffd_psnr = round(peak_signal_noise_ratio(img5,img1),2)
deam_psnr = round(peak_signal_noise_ratio(img5,img2),2)
nb_psnr = round(peak_signal_noise_ratio(img5,img3),2)
tc_psnr = round(peak_signal_noise_ratio(img5,img4),2)


plt.subplot(1,6,1)
plt.imshow(img6)
plt.axis('off')
plt.title("\n"+"Noisy",y=-0.22,fontsize=18)
plt.subplot(1,6,2)
plt.imshow(img1)
plt.axis('off')
plt.title(str(ffd_psnr)+"dB\n"+"FFDNet",y=-0.22,fontsize=18)
plt.subplot(1,6,3)
plt.imshow(img2)
plt.axis('off')
plt.title(str(deam_psnr)+"dB\n"+"DeamNet",y=-0.22,fontsize=18)
plt.subplot(1,6,4)
plt.imshow(img3)
plt.axis('off')
plt.title(str(nb_psnr)+"dB\n"+"NBNet",y=-0.22,fontsize=18)
plt.subplot(1,6,5)
plt.imshow(img4)
plt.axis('off')
plt.title(str(tc_psnr)+"dB\n"+"Ours",y=-0.22,fontsize=18)
plt.subplot(1,6,6)
plt.imshow(img5)
plt.axis('off')
plt.title("\n"+"GT",y=-0.22,fontsize=18)
plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.01, hspace=0.01)
plt.show()



