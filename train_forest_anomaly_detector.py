from rgbhistogram import RGBHistogram
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import argparse
import pickle
import cv2

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=False, default='./intro-anomaly-detection/forest', help="path to dataset of training images")
    ap.add_argument("--model", required=False, default='./models/forest_anomoly_detector.model', help="path/name to store models")

    args = vars(ap.parse_args())

    print(f"Loading dataset")
    dataset, _ = RGBHistogram.load_dataset(args['dataset'], include_color_stats=True, bins=(3,3,3), color_cvt=cv2.COLOR_BGR2HSV)


    # train the anomly detection model
    print("Fitting anomoly detection model")
    model = IsolationForest(n_estimators=100, contamination=0, behaviour="new", random_state=42)
    # model = LocalOutlierFactor(n_neighbors=10, novelty=True)
    model.fit(dataset)

    print("Save model to disk")
    with open(args['model'], "wb") as f:
        f.write(pickle.dumps(model))

    print("Done")

