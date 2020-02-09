import cv2
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer

img_rgb = cv2.imread('x5.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
star = cv2.imread('template2.png', 0)
star_w, star_h = star.shape[::-1]

#drow = cv2.imread('chars/drow.png', 0)

def run(template, name):
    w, h = template.shape[::-1]
    w = w + 10
    h = h + 15

    start = timer()
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    #print('Found:', len(loc[0]))
    boxes = []
    np_images = []
    for pt in zip(*loc[::-1]):
        bb = {}
        bb['x1'] = x = pt[0]
        bb['x2'] = x2 = pt[0] + w
        bb['y1'] = y = pt[1]
        bb['y2'] = y2 = pt[1] + h
        crop = img_rgb[y:y2, x:x2]

        if len(boxes) > 0:
            found_overlap = False
            for box in boxes:
                overlap_percent = get_iou(box, bb)
                if(overlap_percent > 0):
                    found_overlap = True
            if found_overlap:
                continue # proceed to next box, overlaps don't get added

        # this only runs if it is the first box or not overlapping
        boxes.append(bb)
        np_images.append(crop)
        cv2.rectangle(img_rgb, pt, (x2, y2), (0, 0, 255), 1)

        # crop image
        im = img_rgb[y:y2, x:x2].copy()
        tier1 = [178, 157, 142]
        tier2 = [184, 198, 221]
        tier3 = [255, 251, 17]

        pix1 = im[33,11][::-1]
        dif1 = pix1-tier1
        dif_sum1 = abs(sum(dif1))
        print('Tier1:', dif_sum1)

        pix2 = im[34,18][::-1]
        dif2 = pix2-tier2
        dif_sum2 = abs(sum(dif2))
        print('Tier2:', dif_sum2)

        pix3 = im[34,17][::-1]
        if(pix3[0] > 250):
            dif_sum3 = 0 # its obviously a tier 3
        else:
            dif_sum3 = 255 # its not a tier 3
        #dif3 = pix3-tier3
        #dif_sum3 = abs(sum(dif3))
        print('Tier3:', dif_sum3)

        evaluated_tier = np.argmin([dif_sum1, dif_sum2, dif_sum3])+1
        print('Min: Tier', evaluated_tier)

        # display crop
        #cv2.imshow('Img', im)
        #cv2.waitKey(0)
        cv2.imwrite(f'{evaluated_tier}/{x}{x2}.png', im)


    #for img in np_images:
    #    #print(img)
    #    pass

    end = timer()
    #print(f'Took: {end-start} seconds')
    #print(f'{name}: {len(boxes)}')

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


if __name__ == '__main__':
    start = timer()
    run(star, 'Drow Ranger')

    #for i in range(60):
    #    run()
    end = timer()
    print(f'Took: {end-start} seconds')

    cv2.imwrite('res.png', img_rgb)
