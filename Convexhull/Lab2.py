import cv2
import matplotlib.pyplot as plt
import numpy as np 
import sys 

#Read Image
img = cv2.imread(
    "/Users/richard/neural-networks-and-image-analysis/Convexhull/input/sample.jpg"
)
gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.blur(gray_image, (3, 3))

#cv2.imshow("Original Image",img)
# cv2.imshow("Gray Image",gray_image)
#cv2.imshow("Blurred Image",blur) 
# cv2.waitKey(0)
# cv2.destroyAllWindows()     

#Binary Thresholding
_, thresh = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
# cv2.imshow("Thresholded Image",thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)
coutours_img =img.copy()
cv2.drawContours(coutours_img, contours, -1, (0,255,0), 3)
# cv2.imshow("Contours",coutours_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows() 

#Create hull array for convexHull points
hull = []       
#Calculate points for each contour
for i in range(len(contours)):
    hull.append(cv2.convexHull(contours[i], False)) 

# create an empty black image
drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
coutours_img_with_hull =coutours_img.copy()
# draw contours and hull points
for i in range(len(contours)):
    color_contours = (0, 255, 0)  
    color = (255, 255, 255) 
    # draw ith contour
    cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
    # draw ith convex hull object
    cv2.drawContours(drawing, hull, i, color, 1, 8)

    cv2.drawContours(coutours_img_with_hull, contours, i, color_contours, 1, 8, hierarchy)
    # draw ith convex hull object
    cv2.drawContours(coutours_img_with_hull, hull, i, (0,0,0), 1, 8)

# cv2.imshow("Convex Hull", drawing)  
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#COnvert BGR->RGB for matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
blurred_rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)
coutours_img_rgb = cv2.cvtColor(coutours_img, cv2.COLOR_BGR2RGB)
drawing_rgb = cv2.cvtColor(drawing, cv2.COLOR_BGR2RGB)
coutours_img_with_hull_rgb = cv2.cvtColor(coutours_img_with_hull, cv2.COLOR_BGR2RGB)


#Display the results
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
axes[0,0].imshow(img_rgb)
axes[0,0].set_title(" Original Image")
axes[0,0].axis("off")           

axes[0,1].imshow(blurred_rgb)
axes[0,1].set_title(" Blurred Image")
axes[0,1].axis("off")       

axes[0,2].imshow(thresh_rgb)
axes[0,2].set_title(" Thresholded Image")
axes[0,2].axis("off")  

axes[1,0].imshow(coutours_img_rgb)
axes[1,0].set_title(" Contours on Image")
axes[1,0].axis("off")   

axes[1,1].imshow(coutours_img_with_hull_rgb)
axes[1,1].set_title(" Contours with Convex Hull on Image")
axes[1,1].axis("off")   

axes[1,2].imshow(drawing_rgb)
axes[1,2].set_title(" Convex Hull")
axes[1,2].axis("off")

plt.tight_layout()
plt.show()
fig.savefig("Convex_Hull.png", dpi=200, bbox_inches="tight")