import argparse
from rgbhistogram import RGBHistogram
import pickle
import cv2
from path_utils import list_images

verbose = 1

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=False, default="./models/forest_anomoly_detector.model",
                    help="path to trained anomaly detection model")
    ap.add_argument("-i", "--dataset_root", required=False, default="/Users/patrickryan/Development/python/mygithub/pyimagesearch-python-machine-learning/3scenes",
                    help="path to input image")
    args = vars(ap.parse_args())

    # load the anomaly detection model
    print("[INFO] loading anomaly detection model...")
    model = pickle.loads(open(args["model"], "rb").read())

    histo = RGBHistogram(bins=(3,3,3), include_color_stats=True, color_cvt=cv2.COLOR_BGR2HSV)

    imagePaths = list_images(args["dataset_root"])
    image_count = 0
    correct_count = 0
    for imagePath in imagePaths:
        image_count += 1

        features, image = histo.get_features(imagePath)

        pred = model.predict([features])[0]
        if 'forest' in imagePath:
            if pred == 1:
                correct_count += 1
        else:
            if pred == -1:
                correct_count += 1


    print(f"Anomly Detection Accuracy: {correct_count/image_count} from a total image dataset of {image_count} images")

