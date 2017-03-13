import numpy as np
import glob
import cv2

from skimage.io import imread
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from scipy.ndimage.measurements import label
from sklearn.externals import joblib
import pickle
from moviepy.editor import VideoFileClip

# Setup global variables
VEHICLE_IMAGE_LOCATION = "data/vehicles/*.png"
NON_VEHICLE_IMAGE_LOCATION = "data/non-vehicles/*.png"
HEATMAP_THRESHOLD = 2
COOL_RATE = 0.9
IMG_SHAPE = (720, 1280)

class VehicleDetector:
    def __init__(self):
        self.clf = None
        self.heatmap = np.zeros(shape=(IMG_SHAPE[0], IMG_SHAPE[1]))

        self.window_bands = self.create_window_bands()

        # Setup HOG descriptor:
        winSize = (64, 64)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        nlevels = 64
        self.hog_descriptor = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    def set_model(self, clf):
        self.clf = clf

    def perform_hog_extraction(self,
                               image):
        """
        Runs HOG feature extraction on an image.
        """
        return np.ravel(self.hog_descriptor.compute(image))

    # Define a function to compute binned color features (from udacity lectures).
    def bin_spatial(self, img, size=(32, 32)):
        return cv2.resize(img, size).ravel()

    # Define a function to compute color histogram features (from udacity lectures).
    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        """
        Computes histogram features on each channel in the image, returning a single flat array of features.
        """
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        return np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    def extract_features_from_image(self, image):
        """
        This method runs HOG only, because bin spatial and color histograms didn't add enough accuracy to justify the
        number of features.
        """
        return self.perform_hog_extraction(image)

    def load_images_and_extract_features(self,
                                         vehicle_images=glob.glob(VEHICLE_IMAGE_LOCATION),
                                         non_vehicle_images=glob.glob(NON_VEHICLE_IMAGE_LOCATION),
                                         cspace="RGB"):
        """
        Loads images and extracts the features, returning X and Y arrays.
        :param vehicle_images: Glob to find the vehicle images.
        :param non_vehicle_images: Glob to find the non-vehicle images.
        :param cspace: If not using RGB, supply a different color space.
        :return:
        """

        print("Loading images and extracting features...")

        Y = []
        X = []

        for i in range(0, 2):
            for image_uri in (non_vehicle_images if i == 0 else vehicle_images):
                image = imread(image_uri)

                # apply color conversion if other than 'RGB'
                if cspace != 'RGB':
                    if cspace == 'HSV':
                        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                    elif cspace == 'LUV':
                        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                    elif cspace == 'HLS':
                        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                    elif cspace == 'YUV':
                        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                else:
                    feature_image = np.copy(image)

                h = self.extract_features_from_image(np.resize(image, (64, 64, 3)))
                Y.append("vehicle" if i == 1 else "non-vehicle")
                X.append(h.astype(np.float64))

                #             image_pyramid = pyramid_gaussian(feature_image, max_layer=0)

                #             for layer in image_pyramid:

        Y = np.array(Y)
        X = np.array(X)

        print("Calculated features on {0} images".format(len(X)))

        return X, Y

    def create_classifier(self):
        return Pipeline([
            ('scaling', StandardScaler(with_mean=0, with_std=1)),
            ('feature_selection', SelectFromModel(ExtraTreesClassifier())),
            ('classification', SVC(kernel="rbf", verbose=1, probability=True))
        ])

    def add_heat(self, bbox_list):
        """
        Adds heat based on bounding boxes and applies cooling.
        :param bbox_list: List of bounding boxes where cars were located.
        :return:
        """
        # frame_heatmap = np.zeros_like(self.heatmap)
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            self.heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
            # frame_heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1


        # Apply cooling
        self.heatmap *= COOL_RATE

        # Threshold heatmap
        self.heatmap[self.heatmap <= HEATMAP_THRESHOLD] = 0

    def slide_window(self, x_start_stop=[None, None], y_start_stop=[None, None],
                     xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = IMG_SHAPE[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = IMG_SHAPE[0]
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]

        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_windows = np.int(xspan / nx_pix_per_step)
        ny_windows = np.int(yspan / ny_pix_per_step)
        # Initialize a list to append window positions to
        window_list = []

        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs * nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys * ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    def create_window_bands(self):
        """
        Programmatically creates the window bands using some tuned parameters.
        :return:
        """
        bottom = 650
        right = 1280
        size = 350
        window_size_step = 10
        bottom_step_size = int((window_size_step / 2) + (window_size_step / 8))
        min_windows_size = 64

        squeeze = 5
        squeeze_inc = 5

        window_bands = []

        while size > min_windows_size:
            #     draw_rect(img, bottom, right, size)
            #     window_dims.append((bottom, size))
            windows = self.slide_window(x_start_stop=(squeeze, IMG_SHAPE[1] - squeeze),
                                        y_start_stop=(bottom - size, bottom),
                                        xy_window=(size, size),
                                        xy_overlap=(0.25, 0.0))

            window_bands.append(windows)
            size = size - window_size_step
            bottom = bottom - bottom_step_size
            right = right - window_size_step * 2
            squeeze += squeeze_inc

        return window_bands

    def detect(self, image):
        """
        Captures 64x64 images from windows and does batch classification.
        :param image:
        :return:
        """
        window_frames = []
        bounding_boxes = []
        for windows in self.window_bands:
            for window in windows:
                left = int(window[0][0])
                right = int(window[1][0])
                top = int(window[0][1])
                bottom = int(window[1][1])

                if left < 0 or top < 0 or bottom > image.shape[0] or right > image.shape[1]:
                    continue

                w = image[top:bottom, left:right]

                w = cv2.resize(w, (64, 64))
                features = self.extract_features_from_image(w)

                window_frames.append([((left, top), (right, bottom)), features])

        window_frames = np.array(window_frames)
        predictions = self.clf.predict(np.vstack(window_frames[:, 1]))

        return window_frames[predictions == 'vehicle'][:, 0]

    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        # Return the image
        return img

    def process_frame(self, frame):
        bounding_boxes = self.detect(frame)
        self.add_heat(bounding_boxes)
        labels = label(self.heatmap)
        return self.draw_labeled_bboxes(frame, labels)


    def process_video(self, video_in, video_out):
        output = video_out
        clip1 = VideoFileClip(video_in)
        white_clip = clip1.fl_image(self.process_frame)  # NOTE: this function expects color images!!
        white_clip.write_videofile(output, audio=False)

def train():
    vehicle_detector= VehicleDetector()

    # Gather the data and split into train and test data.
    X, Y = vehicle_detector.load_images_and_extract_features()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)



    # Create train the classifier
    clf = vehicle_detector.create_classifier()

    print("Fitting SVM")
    clf.fit(x_train, y_train)
    acc = accuracy_score(clf.predict(x_test), y_test)
    print("SVM accuracy: {0}".format(acc))

    # Save the model and data
    joblib.dump(clf, 'svm.p')
    pickle.dump((x_train, x_test, y_train, y_test), open("xxyy.p", "wb"))

def main():
    # Load the model
    clf = joblib.load("svm.p")
    vehicle_detector = VehicleDetector()
    vehicle_detector.set_model(clf)

    vehicle_detector.process_video("project_video.mp4", "project_video_out.mp4")


if __name__ == "__main__":
    train()
    main()
