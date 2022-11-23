
def sma(l, period):
    total = np.sum(l[-period:])
    return total/period


def roi_from_lines(lines):
    # lines = lines.squeeze()
    roi_points = []

    for line in lines:
        line = line.squeeze()
        roi_points.append(line[:2])
        roi_points.append(line[2:])

    return np.array(roi_points, dtype=np.int32)

def show_image(name, img):  # function for displaying the image
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_canny(img, thresh_low, thresh_high):  # function for implementing the canny
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # show_image('gray',img_gray)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # show_image('blur',img_blur)
    img_canny = cv2.Canny(img_blur, thresh_low, thresh_high)
    # show_image('Canny', img_canny)
    return img_canny


def region_of_interest(image, bounds):  # function for extracting region of interest
    # bounds in (x,y) format

    bounds = bounds.reshape((1, -1, 2))
    # bounds = np.array([[[0,image.shape[0]],[0,image.shape[0]/2],[900,image.shape[0]/2],[900,image.shape[0]]]],dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, bounds, [255, 255, 255])
    # show_image('inputmask',mask)
    masked_image = cv2.bitwise_and(image, mask)
    # show_image('mask', masked_image)
    return masked_image, mask


def draw_lines(img, lines):  # function for drawing lines on black mask
    mask_lines = np.zeros_like(img)
    for points in lines:
        x1, y1, x2, y2 = points[0]
        cv2.line(mask_lines, (x1, y1), (x2, y2), [0, 0, 255], 2)

    return mask_lines


def get_coordinates(img, line_parameters):  # functions for getting final coordinates
    slope = line_parameters[0]
    intercept = line_parameters[1]
    y1 = 300
    y2 = 120
    # y1=img.shape[0]
    # y2 = 0.6*img.shape[0]
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [x1, int(y1), x2, int(y2)]