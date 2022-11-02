import cv2
import numpy as np
import matplotlib.pyplot as plt


def WipeOut(image):
    rows, cols = image.shape
    rows_half, cols_half = int(rows / 2), int(cols / 2)
    rows_quat, cols_quat = int(rows / 4), int(cols / 4)
    rows_spe, cols_spe = int(rows / 12), int(cols / 12)
    mask = np.zeros((rows, cols, 2), dtype=np.uint8)
    field1 = 20
    field2 = 15
    field3 = 4

    # main points
    mask[rows_quat - field1:rows_quat + field1, cols_half - field1:cols_half + field1] = 1
    mask[3 * rows_quat - field1:3 * rows_quat + field1, cols_half - field1:cols_half + field1] = 1
    mask[rows_half - field1:rows_half + field1, 3 * cols_quat - field1:3 * cols_quat + field1] = 1
    mask[rows_half - field1:rows_half + field1, cols_quat - field1:cols_quat + field1] = 1

    # Lines
    mask[rows_quat - field3:rows_quat + field3, 0:cols] = 1
    mask[3 * rows_quat - field3: 3 * rows_quat + field3, 0:cols] = 1
    mask[0:rows, cols_quat - 2 * field3:cols_quat + 2 * field3] = 1
    mask[0:rows, 3 * cols_quat - 2 * field3:3 * cols_quat + 2 * field3] = 1
    mask[0:rows, cols_spe - field3:cols_spe + field3] = 1
    mask[0:rows, cols - cols_spe - field3:cols - cols_spe + field3] = 1

    # Cornel
    mask[rows_spe - field2:rows_spe + field2, cols_half - field2:cols_half + field2] = 1
    mask[rows - rows_spe - field2:rows - rows_spe + field2, cols_half - field2:cols_half + field2] = 1
    mask[rows_half - field2:rows_half + field2, cols_spe - field2:cols_spe + field2] = 1
    mask[rows_half - field2:rows_half + field2, cols - cols_spe - field2:cols - cols_spe + field2] = 1

    # Inside
    mask[rows_half - rows_spe - field2:rows_half - rows_spe + field2, cols_half - field2:cols_half + field2] = 1
    mask[rows_half + rows_spe - field2:rows_half + rows_spe + field2, cols_half - field2:cols_half + field2] = 1
    mask[rows_half - field2:rows_half + field2, cols_half + cols_spe - field2:cols_half + cols_spe + field2] = 1
    mask[rows_half - field2:rows_half + field2, cols_half - cols_spe - field2:cols_half - cols_spe + field2] = 1
    return mask

    # mask[rows_half - 30:rows_half + 30, cols_half - 30:cols_half + 30] = 1


def main():
    img = cv2.imread('lisa_trans.png')

    # Turn it grey
    img = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]

    # Fourier transform using cv2
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dftShift = np.fft.fftshift(dft)
    result = 20 * np.log(
        cv2.magnitude(dftShift[:, :, 0], dftShift[:, :, 1]))  # Reverse between the middle and the corner

    mask = WipeOut(img)

    fShift = dftShift * mask
    ishift = np.fft.ifftshift(fShift)
    result_trans = 20 * np.log(cv2.magnitude(fShift[:, :, 0], fShift[:, :, 1]))
    iimg = cv2.idft(ishift)
    iimg = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])

    plt.subplot(221)
    plt.imshow(img, cmap="gray")
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(result, cmap="gray")
    plt.subplot(223)
    plt.imshow(iimg, cmap="gray")
    plt.subplot(224)
    plt.imshow(result_trans, cmap="gray")
    plt.savefig('hah.png', bbox_inches='tight', pad_inches=0.0)
    # plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
