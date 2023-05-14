import cv2
import numpy as np
import os
from skimage.feature import hog
from skimage import exposure
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# def remove_shadows(image):
#     # convert to LAB color space
#     # print('image', image.dtype)
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        

#     # split the LAB channels
#     l, a, b = cv2.split(lab)
    
#     # apply the Retinex algorithm to the L channel
#     # set the gain to 128 for best results
#     l = cv2.createCLAHE(clipLimit=1, tileGridSize=(3,3)).apply(l)
#     # merge the LAB channels back together
#     lab = cv2.merge((l,a,b))

#     # convert back to BGR color space
#     result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
#     return result


men_image_dir = "../Sign-Language-Recognition/dataset_sample/men/"
women_image_dir = "../Sign-Language-Recognition/dataset_sample/women/"
# bgr_img = cv2.imread(image_dir)
# cv2.imwrite('../Youssef tomfoolery/orgin.png',bgr_img)



# hsv_img = cv2.cvtColor(bgr_img,cv2.COLOR_BGR2HSV)
# cv2.imwrite('../Youssef tomfoolery/hsv.png',hsv_img)


light_skin = np.array([7, 50, 50],np.uint8)
dark_skin = np.array([15, 255, 255],np.uint8)


# frame_threshed = cv2.inRange(hsv_img, light_skin, dark_skin)
# cv2.imwrite('../Youssef tomfoolery/threshed.png', frame_threshed)

# denoised_img = cv2.fastNlMeansDenoising(frame_threshed,None,15,7,21)
# cv2.imwrite('../Youssef tomfoolery/output.png', denoised_img)
men_images = []
men_labels = []
for i in range(0,6):
    for filename in os.listdir(men_image_dir + "/" + str(i)):
        img = cv2.imread(os.path.join(men_image_dir + "/" + str(i),filename))
        if img is not None:
            men_images.append(img)
            match (filename[0]):
                case "0":
                    men_labels.append(0)
                case "1":
                    men_labels.append(1)
                case "2":
                    men_labels.append(2)
                case "3":
                    men_labels.append(3)
                case "4":
                    men_labels.append(4)
                case "5":
                    men_labels.append(5)

women_images = []
women_labels = []
for i in range(0,6):
    for filename in os.listdir(women_image_dir + "/" + str(i)):
        img = cv2.imread(os.path.join(women_image_dir + "/" + str(i),filename))
        if img is not None:
            women_images.append(img)
            match (filename[0]):
                case "0":
                    women_labels.append(0)
                case "1":
                    women_labels.append(1)
                case "2":
                    women_labels.append(2)
                case "3":
                    women_labels.append(3)
                case "4":
                    women_labels.append(4)
                case "5":
                    women_labels.append(5)
men_samples = [(men_labels[i],men_images[i]) for i in range(0,len(men_labels))]
women_samples = [(women_labels[i],women_images[i]) for i in range(0,len(women_labels))]

full_samples = men_samples + women_samples
full_samples = sorted(full_samples, key = lambda x: x[0])

sample_input = [sample[1] for sample in full_samples]
sample_target = [sample[0] for sample in full_samples]
print("done")

# bgr_input = [cv2.cvtColor(s_input,cv2.COLOR_RGB2BGR) for s_input in sample_input]
# #hsv_input = [cv2.cvtColor(b_input,cv2.COLOR_BGR2HSV) for b_input in bgr_input]

resized_input = [cv2.resize(s_input,(200,200), interpolation= cv2.INTER_AREA) for s_input in sample_input]

blurred_input = [ cv2.medianBlur(r_input,5) for r_input in resized_input]

gray_input = [cv2.cvtColor(b_input,cv2.COLOR_BGR2GRAY) for b_input in blurred_input]



# Define the lower and upper skin color bounds in YCrCb color space
light_skin = np.array([7, 50, 50],np.uint8)
dark_skin = np.array([15, 255, 255],np.uint8)
X_processed = []

# for i in range(len(sample_input)):
#     img = sample_input[i]
#     img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred_input = cv2.medianBlur(gray,5)
#     X_processed.append(blurred_input)
    
# X_processed = np.array(X_processed)
print("done2")

# for i in range(0,len(X_train_processed)):
#     cv2.imwrite("./Stage1_output/" + str(i) + ".png",X_train_processed[i])



feature_descriptor = []
hog_imgs = []

for i in range(0,len(gray_input)):
    feature_descriptor_i = hog(gray_input[i], orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=False)
    feature_descriptor.append(feature_descriptor_i)


print('done3')
x_fit, x_test, y_fit, y_test = train_test_split(feature_descriptor, sample_target, test_size=0.25)


clf = SVC(kernel='rbf',C=100,gamma=0.001)
clf = clf.fit(x_fit,y_fit)

pred_y = clf.predict(x_test)

accuracy = accuracy_score(y_test,pred_y)

print('Accuracy: {} % '. format(accuracy * 100))

params_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
          'gamma': [0.0001, 0.001, 0.01, 0.1],
          'kernel':['linear','rbf'] }

#Create the GridSearchCV object
grid_clf = GridSearchCV(SVC(class_weight='balanced'), params_grid,verbose=3,refit=True)

#Fit the data with the best possible parameters
grid_clf = grid_clf.fit(x_fit, y_fit)

#Print the best estimator with it's parameters
print(grid_clf.best_estimator_)