import numpy as np
from skimage import transform


def resize_min(patch, min_size):
    """
    Resizes the minimum dimension of an image to a new size preserving the
    aspect ratio
    :param patch: Numpy array. The image to resize.
    :param min_size: Positive integer. New size of the minimum image dimension
    :return: Resized numpy array
    """
    if min_size is None:
        raise ValueError('min_size must be provided as a positive integer.')
    patch_size = list(patch.shape)
    if len(patch_size) - 3 == 0:
        patch_size[2] = []
    aspect_ratio = np.float(patch_size[0]) / np.float(patch_size[1])
    if patch_size[0] - patch_size[1] < 0:
        new_row = min_size
        new_col = np.int(np.float(new_row) / aspect_ratio)
    else:
        new_row = np.int(np.float(min_size) * aspect_ratio)
        new_col = min_size
    patch_new = transform.resize(patch, (new_row, new_col), order=3)
    return patch_new/255


def resize_max(patch, max_size):
    """
    Resizes the maximum dimension of an image to a new size preserving the
    aspect ratio.
    :param patch: Numpy array. The image to resize.
    :param max_size: Positive integer. New size of the maximum image dimension
    :return: Resized numpy array
    """
    if max_size is None:
        raise ValueError('max_size must be provided as a positive integer.')
    patch_size = list(patch.shape)
    if len(patch_size) - 3 == 0:
        patch_size[2] = []
    aspect_ratio = np.float(patch_size[0]) / np.float(patch_size[1])
    if patch_size[0] - patch_size[1] < 0:
        new_row = np.int(np.float(max_size) * aspect_ratio)
        new_col = max_size
    else:
        new_row = max_size
        new_col = np.int(np.float(new_row) / aspect_ratio)
    patch_new = transform.resize(patch, (new_row, new_col), order=3)
    return patch_new/255


def resize_exact(patch, new_size):
    """
    Resizes an image to a new size without any regard to the aspect ratio.
    :param patch: Numpy array. The image to resize.
    :param new_size: Length-2 tuple of positive integers. New size of the image
    :return: Resized numpy array
    """
    if len(new_size) - 2 != 0:
        raise ValueError('new_size must be provided as a tuple of length 2.')

    patch_new = transform.resize(patch, new_size, order=3)
    return patch_new/255


def flip_patch(patch):
    """
    Creates a mirror image of the input image.
    :param patch: Numpy array. The input image
    :return: Numpy array. A mirror image of the input image.
    """
    new_patch = np.zeros_like(patch)
    if len(patch.shape) == 3:
        for col in range(patch.shape[1]):
            new_patch[:, col, :] = patch[:, patch.shape[1] - col - 1, :]
    else:
        for col in range(patch.shape[1]):
            new_patch[:, col] = patch[:, patch.shape[1] - col - 1]

    return new_patch/255


def crop_random_tl(patch, crop_size):
    """
    Randomly extracts a patch of size `crop_size` from the input image such
    that the patch covers the top-left portion of the image
    :param patch: Numpy array. Input image to extract patch from.
    :param crop_size: A length-2 tuple of positive integers. Size of the
    extracted patch
    :return: Numpy array. Extracted patch
    """
    if crop_size is None:
        raise ValueError('crop_size must be provided as a tuple of length 2.')

    patch_size = patch.shape
    if patch_size < crop_size:
        raise ValueError(
            'Some or all of crop dimensions are larger than the patch '
            'dimensions.')

    rowbegin = np.random.randint(low=0, high=patch_size[0] - crop_size[0] + 1)
    colbegin = np.random.randint(low=0, high=patch_size[1] - crop_size[1] + 1)
    return patch[rowbegin: rowbegin + crop_size[0],
           colbegin: colbegin + crop_size[1]]/255


def crop_random_tr(patch, crop_size):
    """
    Randomly extracts a patch of size `crop_size` from the input image such
    that the patch covers the top-right portion of the image
    :param patch: Numpy array. Input image to extract patch from.
    :param crop_size: A length-2 tuple of positive integers. Size of the
    extracted patch
    :return: Numpy array. Extracted patch
    """
    if crop_size is None:
        raise ValueError('crop_size must be provided as a tuple of length 2.')

    patch_size = patch.shape
    if patch_size < crop_size:
        raise ValueError(
            'Some or all of crop dimensions are larger than the patch '
            'dimensions.')

    rowbegin = np.random.randint(low=0, high=patch_size[0] - crop_size[0])
    colbegin = patch_size[1] - 1 - np.random.randint(low=0, high=patch_size[1] -
                                                                 crop_size[1])
    return patch[rowbegin: rowbegin + crop_size[0],
           colbegin - crop_size[1]: colbegin]/255


def crop_random_bl(patch, crop_size):
    """
    Randomly extracts a patch of size `crop_size` from the input image such
    that the patch covers the bottom-left portion of the image
    :param patch: Numpy array. Input image to extract patch from.
    :param crop_size: A length-2 tuple of positive integers. Size of the
    extracted patch
    :return: Numpy array. Extracted patch
    """
    if crop_size is None:
        raise ValueError('crop_size must be provided as a tuple of length 2.')

    patch_size = patch.shape
    if patch_size < crop_size:
        raise ValueError(
            'Some or all of crop dimensions are larger than the patch '
            'dimensions.')

    rowbegin = patch_size[0] - 1 - np.random.randint(low=0, high=patch_size[0] -
                                                                 crop_size[0])
    colbegin = np.random.randint(low=0, high=patch_size[1] - crop_size[1])
    return patch[rowbegin - crop_size[0]: rowbegin,
           colbegin: colbegin + crop_size[1]]/255


def crop_random_br(patch, crop_size):
    """
    Randomly extracts a patch of size `crop_size` from the input image such
    that the patch covers the bottom-right portion of the image
    :param patch: Numpy array. Input image to extract patch from.
    :param crop_size: A length-2 tuple of positive integers. Size of the
    extracted patch
    :return: Numpy array. Extracted patch
    """
    if crop_size is None:
        raise ValueError('crop_size must be provided as a tuple of length 2.')

    patch_size = patch.shape
    if patch_size < crop_size:
        raise ValueError(
            'Some or all of crop dimensions are larger than the patch '
            'dimensions.')

    rowbegin = patch_size[0] - 1 - np.random.randint(low=0, high=patch_size[0] -
                                                                 crop_size[0])
    colbegin = patch_size[1] - 1 - np.random.randint(low=0, high=patch_size[1] -
                                                                 crop_size[1])
    return patch[rowbegin - crop_size[0]: rowbegin,
           colbegin - crop_size[1]: colbegin]/255


def crop_center(patch, crop_size):
    """
    Randomly extracts a patch of size `crop_size` from the input image such
    that the patch covers the center of the image
    :param patch: Numpy array. Input image to extract patch from.
    :param crop_size: A length-2 tuple of positive integers. Size of the
    extracted patch
    :return: Numpy array. Extracted patch
    """
    if crop_size is None:
        raise ValueError('crop_size must be provided as a tuple of length 2.')

    patch_size = map(lambda x: np.ceil(np.float(x) / 2), patch.shape)
    patch_size = list(patch_size)
    crop_size = map(lambda x: np.ceil(np.float(x) / 2), crop_size)
    crop_size = list(crop_size)
    if patch_size < crop_size:
        raise ValueError(
            'Some or all of crop dimensions are larger than the patch '
            'dimensions.')

    return patch[np.int(np.ceil(patch_size[0] - crop_size[0])): np.int(
        np.ceil(patch_size[0] + crop_size[0])) - 1,
           np.int(np.ceil(patch_size[1] - crop_size[1])): np.int(
               np.ceil(patch_size[1] + crop_size[1])) - 1]/255
