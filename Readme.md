# PyImageSearch Intro to Anomly Detection

This repo contains my work while working through the blog post on PyImageSearch.com

You can find that post [here](https://www.pyimagesearch.com/2020/01/20/intro-to-anomaly-detection-with-opencv-computer-vision-and-scikit-learn/)


## My Changes

### RGBHistogram class

I created a class called RGBHistogram that will extract color/tonal characteristics from an Image.

The returned feature set is normalized values for:

[mean(Red), mean(Green), mean(Blue), std(Red), std(Green), std(Blue)]

along with the histogram of color channels 

### IsolationForest

I found the IsolationForest to be very sensitive to the bin size.  3,3,3 seemed to be the best.

### Scoring

I created a script to score the approach using the 3scenes dataset described in this blog post:

https://www.pyimagesearch.com/2019/01/14/machine-learning-in-python/]

I found an overal Anomoly Detection Accuracy of 63.5%.


