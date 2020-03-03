import argparse
from rgbhistogram import RGBHistogram
import pickle
import cv2

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=False, default="./models/forest_anomoly_detector.model",
                    help="path to trained anomaly detection model")
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    args = vars(ap.parse_args())

    # load the anomaly detection model
    print("[INFO] loading anomaly detection model...")
    model = pickle.loads(open(args["model"], "rb").read())

    histo = RGBHistogram(bins=(3,3,3), include_color_stats=True, color_cvt=cv2.COLOR_BGR2HSV)
    features, image = histo.get_features(args['image'])

    pred = model.predict([features])[0]
    label = 'anomly' if pred == -1 else 'normal'
    color = (0,0,255) if pred == -1 else (0, 255, 0)

    cv2.putText(image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Detector", image)
    cv2.waitKey(0)

