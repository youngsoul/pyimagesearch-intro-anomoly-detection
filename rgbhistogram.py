import cv2
import numpy as np
from imutils import paths
"""
This histogram will be used to charac- terize the color of the flower petals, 
which is a good starting point for classifying the species of a flower
"""
from sklearn.preprocessing import MinMaxScaler

class RGBHistogram:

    def __init__(self, bins, include_color_stats=True, color_cvt=None):
        self.bins = bins
        self.color_cvt = color_cvt
        self.include_color_stats = include_color_stats

    def get_features(self, imagePath):
        image = cv2.imread(imagePath)
        img_copy = image.copy()
        if self.color_cvt:
            image = cv2.cvtColor(image, self.color_cvt)

        features = []
        if image is not None:
            if self.include_color_stats:
                features.extend(self.extract_color_stats(image))
            features.extend(self.describe(image).tolist())

        return features, img_copy

    def extract_color_stats(self, image):
        # split the input image into its respective RGB color channels
        # and then create a feature vector with 6 values: the mean and
        # standard deviation for each of the 3 channels, respectively
        (R,G,B) = cv2.split(image)  # depends on the colorcvt, it might be HSV but it does not really matter

        means = [np.mean(R)/np.max(R), np.mean(G)/np.max(G), np.mean(B)/np.max(B)]
        stds = [np.std(R)/np.max(R),np.std(G)/np.max(G), np.std(B)/np.max(B)]
        stats = []
        stats.extend(means)
        stats.extend(stds)

        # return our set of features
        return stats

    def describe(self, image, mask=None):
        hist = cv2.calcHist([image], [0, 1, 2],
                            mask, self.bins, [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)

        # return as a feature vector
        return hist.flatten()

    @staticmethod
    def load_dataset(datasetPath, bins, include_color_stats=True, color_cvt=None):
        histo = RGBHistogram(bins=bins, include_color_stats=include_color_stats, color_cvt=color_cvt)

        # grab the paths to all images in our dataset directory, then
        # initialize our lists of images
        imagePaths = list(paths.list_images(datasetPath))
        data = []

        # loop over the image paths
        for imagePath in imagePaths:
            # quantify the image and update the data list
            features, _ = histo.get_features(imagePath)

            data.append(features)

        # return our data list as a NumPy array
        return np.array(data), imagePaths


if __name__ == '__main__':
    result, imagePaths = RGBHistogram.load_dataset("./intro-anomaly-detection/forest", (3,3,3), color_cvt=cv2.COLOR_BGR2HSV)

    print(len(result))