# Human-activity-and-energy-emission-calculation

Here's a detailed summary of the project:

## Project Overview

This project is a Human Activity Recognition (HAR) system that uses computer vision and deep learning techniques to detect and recognize human activities in a video. The system extracts frames from a video, detects people in each frame using YOLOv4, and then uses the Blip model for image captioning to generate a caption for each detected person. The caption is then used to infer the action being performed by the person.

## Techniques Used

Computer Vision: OpenCV is used for image and video processing, including frame extraction, object detection, and image resizing.
Object Detection: YOLOv4 is used for detecting people in each frame. YOLOv4 is a real-time object detection system that uses a single neural network to predict bounding boxes and class probabilities directly from full images.
Image Captioning: The Blip model is used for generating captions for each detected person. The Blip model is a transformer-based model that uses a combination of computer vision and natural language processing techniques to generate captions for images.
Deep Learning: The Blip model is a deep learning model that uses a transformer architecture to generate captions for images.

                                      +---------------+
                                      |  Video File  |
                                      +---------------+
                                             |
                                             |
                                             v
                                      +---------------+
                                      |  Frame Extraction  |
                                      |  (OpenCV)          |
                                      +---------------+
                                             |
                                             |
                                             v
                                      +---------------+
                                      |  Object Detection  |
                                      |  (YOLOv4)          |
                                      +---------------+
                                             |
                                             |
                                             v
                                      +---------------+
                                      |  Image Captioning  |
                                      |  (Blip Model)      |
                                      +---------------+
                                             |
                                             |
                                             v
                                      +---------------+
                                      |  Action Recognition  |
                                      |  (Caption Analysis) |
                                      +---------------+
                                             |
                                             |
                                             v
                                      +---------------+
                                      |  Data Storage      |
                                      |  (Frame, Person ID,  |
                                      |   Action)          |
                                      +---------------+
