from PIL import Image
from clab import util
from clab.util import imutil
import torch
import numpy as np

# def accimage_loader(path):
#     """
#     >>> path = ut.grab_test_imgpath('lena.png')
#     """
#     import accimage
#     acc_img = accimage.Image(path)


def np_loader(fpath, colorspace=None):
    im_in = imutil.imread(fpath)
    if colorspace is not None:
        cv_255 = im_in  # Assume we read a (bgr) byte image
        cv_01 = cv_255.astype(np.float32) / 255.0
        n_channels = imutil.get_num_channels(cv_01)
        if n_channels == 1:
            input_space = 'GRAY'
        elif n_channels == 3:
            input_space = 'BGR'
        elif n_channels == 4:
            input_space = 'BGRA'
        else:
            raise NotImplementedError()
        output_space = colorspace.upper()
        if output_space == 'L':
            output_space = 'GRAY'
        im_out = imutil.convert_colorspace(cv_01, dst_space=output_space,
                                           src_space=input_space)
    else:
        im_out = im_in
    return im_out


def pil_loader(fpath, colorspace=None):
    """
    Example:
        >>> from clab.im_loaders import *
        >>> assert int(Image.PILLOW_VERSION.split('.')[0]) >= 4
        >>> fpath = ub.grabdata('http://i.imgur.com/JGrqMnV.png', fname='lena.png')
        >>> pil_img = pil_loader(fpath)
        >>> print('pil_img = {!r}'.format(pil_img))
        >>> fpath = ub.grabdata('http://i.imgur.com/iXNf4Me.png', fname='ada.png')
        >>> pil_img = pil_loader(fpath)
        >>> print('pil_img = {!r}'.format(pil_img))
        >>> fpath = ub.grabdata('https://ghostscript.com/doc/tiff/test/images/rgb-3c-16b.tiff')
        >>> pil_img = pil_loader(fpath)
        >>> print('pil_img = {!r}'.format(pil_img))
        >>> fpath = ub.grabdata('http://www.topcoder.com/contest/problem/UrbanMapper3D/JAX_Tile_043_DTM.tif')
        >>> pil_img = pil_loader(fpath)
        >>> print('pil_img = {!r}'.format(pil_img))

        np.asarray(pil_img.getdata())

        from PIL import Image
        import numpy as np
        import ubelt as ub

        # Grab some test data
        dtm_fpath = ub.grabdata('http://www.topcoder.com/contest/problem/UrbanMapper3D/JAX_Tile_043_DTM.tif')

        # Open the tiff image
        pil_img = Image.open(dtm_fpath)

        # Map PIL mode to numpy dtype (note this may need to be extended)
        dtype = {'F': np.float32, 'L': np.uint8}[pil_img.mode]

        # Load the data into a flat numpy array and reshape
        np_img = np.array(pil_img.getdata(), dtype=dtype)
        w, h = pil_img.size
        np_img.shape = (h, w, np_img.size // (w * h))


    Ignore:
        >>> from clab.im_loaders import *
        >>> import torchvision.transforms.functional as tvf
        >>> tvf.to_tensor(pil_img)
        >>> dtm_fpath = ub.grabdata('http://www.topcoder.com/contest/problem/UrbanMapper3D/JAX_Tile_043_DTM.tif')
        >>> pil_img = Image.open(dtm_fpath)
    """
    # with open(fpath, 'rb') as f:
    #     with Image.open(f) as pil_img:
    pil_img = Image.open(fpath)
    pil_img.load()
    if colorspace and pil_img.mode != colorspace:
        pil_img = pil_img.convert(colorspace)

    # pil_img = Image.open(fpath)
    # pil_img = pil_img.convert('RGB')
    # import cv2
    # np_rgb = np.array(pil_img).astype(np.float32) / 255
    # np_lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
    # img = torch.ByteTensor(torch.ByteStorage.from_buffer(pil_img.tobytes()))
    return pil_img


def to_lab(img):
    if isinstance(img, np.ndarray):
        pass

    from PIL import Image, ImageCms  # NOQA
    if img.mode != 'RGB':
        img = img.convert('RGB')

    srgb_profile = ImageCms.createProfile('sRGB')
    lab_profile  = ImageCms.createProfile('LAB')

    rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(
        srgb_profile, lab_profile, 'RGB', 'LAB')
    lab_img = ImageCms.applyTransform(img, rgb2lab_transform)

    np.asarray(lab_img)
    return lab_img


def pil_image_to_float_tensor(pic):
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def numpy_image_to_float_tensor(im):
    return torch.from_numpy(util.atleast_nd(im, n=3).transpose(2, 0, 1)).float()


def pil_label_to_long_tensor(pic):
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    assert nchannel == 1
    img = img.view(pic.size[1], pic.size[0])
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    # img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.long()
    else:
        return img.long()


def numpy_label_to_long_tensor(gt):
    return torch.from_numpy(gt).long()


def image_to_float_tensor(im):
    if isinstance(im, np.ndarray):
        return numpy_image_to_float_tensor(im)
    else:
        return pil_image_to_float_tensor(im)


def label_to_long_tensor(gt):
    if isinstance(gt, np.ndarray):
        return numpy_label_to_long_tensor(gt)
    else:
        return pil_label_to_long_tensor(gt)


def _ignore():
    """
    import plottool as pt
    import matplotlib as mpl
    pt.qtensure()
    # lab = imutil.convert_colorspace(bgr, 'LAB', src_space='BGR')
    # hsv = imutil.convert_colorspace(bgr, 'HSV', src_space='BGR')

    fpath = task.fullres.im_paths[0]
    bgr = cv2.imread(fpath)
    bgr = imutil.ensure_float01(bgr)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    if True:
        pt.imshow(lab[:, :, 0], pnum=(3, 3, 1), norm=mpl.colors.Normalize(), title='L')
        pt.imshow(lab[:, :, 1], pnum=(3, 3, 2), norm=mpl.colors.Normalize(), title='a')
        pt.imshow(lab[:, :, 2], pnum=(3, 3, 3), norm=mpl.colors.Normalize(), title='b')

        pt.imshow(bgr[:, :, 0], pnum=(3, 3, 4), norm=mpl.colors.Normalize(), title='B')
        pt.imshow(bgr[:, :, 1], pnum=(3, 3, 5), norm=mpl.colors.Normalize(), title='G')
        pt.imshow(bgr[:, :, 2], pnum=(3, 3, 6), norm=mpl.colors.Normalize(), title='R')

        pt.imshow(hsv[:, :, 0], pnum=(3, 3, 7), norm=mpl.colors.Normalize(), title='H')
        pt.imshow(hsv[:, :, 1], pnum=(3, 3, 8), norm=mpl.colors.Normalize(), title='S')
        pt.imshow(hsv[:, :, 2], pnum=(3, 3, 9), norm=mpl.colors.Normalize(), title='V')
    else:
        import colormath
        import logging
        logging.getLogger('colormath.chromatic_adaptation').setLevel(logging.INFO)
        logging.getLogger('colormath.color_conversions').setLevel(logging.INFO)
        logging.getLogger('colormath.color_objects').setLevel(logging.INFO)

        cmap_a_gr = pt.interpolated_colormap([
            (pt.GREEN, 0.0),
            (pt.GRAY, 0.5),
            (pt.RED,   1.0)
        ], resolution=1024, space='lab')
        cmap_b_by = pt.interpolated_colormap([
            (pt.BLUE,   0.0),
            (pt.GRAY,  0.5),
            (pt.YELLOW, 1.0)
        ], resolution=1024, space='lab')

        # hack, not quite real hsv
        # the normal hsv colormap works fine
        # hack_cmap_hue = pt.interpolated_colormap([(pt.RED, 0), ((1, 0, .000001, 0), 1.0)], resolution=1024, space='hsv')

        pt.imshow(bgr, fnum=2)

        pt.figure(fnum=1)
        pt.imshow(lab[:, :, 0], pnum=(3, 3, 1), norm=mpl.colors.Normalize(), title='L')
        pt.imshow(lab[:, :, 1], pnum=(3, 3, 2), norm=mpl.colors.Normalize(vmin=-100, vmax=100), title='a', cmap=cmap_a_gr)
        pt.imshow(lab[:, :, 2], pnum=(3, 3, 3), norm=mpl.colors.Normalize(vmin=-100, vmax=100), title='b', cmap=cmap_b_by)

        pt.imshow(bgr[:, :, 0], pnum=(3, 3, 4), norm=mpl.colors.Normalize(), title='B')
        pt.imshow(bgr[:, :, 1], pnum=(3, 3, 5), norm=mpl.colors.Normalize(), title='G')
        pt.imshow(bgr[:, :, 2], pnum=(3, 3, 6), norm=mpl.colors.Normalize(), title='R')

        pt.imshow(hsv[:, :, 0], pnum=(3, 3, 7), norm=mpl.colors.Normalize(vmin=0, vmax=360), title='H', cmap='hsv')
        pt.imshow(hsv[:, :, 1], pnum=(3, 3, 8), norm=mpl.colors.Normalize(), title='S')
        pt.imshow(hsv[:, :, 2], pnum=(3, 3, 9), norm=mpl.colors.Normalize(), title='V')
    """


def rgb_tensor_to_imgs(tensor_data, norm=False):
    if len(tensor_data.shape) == 4:
        arr = tensor_data.cpu().numpy().transpose(0, 2, 3, 1)
        if arr.shape[3] == 1:
            ims = arr[:, :, :, 0]
        elif arr.shape[3] == 3:
            ims = arr
        else:
            raise ValueError('unexpected')
        if norm:
            extent = (ims.max() - ims.min())
            ims = (ims - ims.min()) / max(extent, 1e-8)
        return ims
    else:
        raise ValueError('unexpected')
