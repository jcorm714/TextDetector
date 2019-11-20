import cv2
import argparse as ap
import os
import numpy as np
import datetime

def detect_text(path):
        
        img = cv2.imread(path)


        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vis = img.copy()
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        hulls = []
        for p in regions:
            for i in range(len(p)):
                hulls.append(cv2.convexHull(p[i].reshape(-1,1,2)))
        cv2.polylines(vis,hulls,1, (0,255,0))

        
        #commented out for batch programming
        #cv2.imshow("image", vis)
        #cv2.waitKey(0)

        mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

        for contour in hulls:
            cv2.drawContours(mask, [contour], -1,(255,255,255),-1)

        text_regions = cv2.bitwise_and(img, img, mask = mask)
        
        #commented out for batch programing
        #cv2.imshow("text", text_regions)
        return text_regions


def save_processed_image(img):
    cwd = os.getcwd()
    img_path = os.path.join(cwd, "processed_images")
    if(not os.path.exists(img_path)):
        os.mkdir(img_path)
    file_name = "img_"
    curr = str(datetime.datetime.now())
    curr = curr.replace(' ', '_')
    curr = curr.replace(':', '_')
    curr = curr.replace(".", '_')
    curr = curr.replace("-", '_')
    file_name += curr + ".png"
    img_path = os.path.join(img_path, file_name)
    cv2.imwrite(img_path, img)


parser = ap.ArgumentParser(description = 'Input data from CLI')
parser.add_argument('--path', type=str, help= "Path directory containining file")

args = parser.parse_args()
print(args)

if(os.path.exists(args.path)):
  
    images = os.listdir(args.path)
    for image in images:
        (root, ext) = os.path.splitext(image)
        if(ext == '.png' or ext == '.jpg'):
            abs_path = os.path.join(args.path, image)
            processed_image = detect_text(abs_path)
            save_processed_image(processed_image)
            cv2.destroyAllWindows()


            
            


