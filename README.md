# Optimized-Warehouse-Management-System
# Introduction


The Optimized Warehouse Management System is designed to revolutionize warehouse operations through the strategic use of artificial intelligence, computer vision, and operational best practices. It focuses on four critical areas to maximize efficiency, safety, and productivity
# Git repo link
https://github.com/IT21822094/Optimized-Warehouse-Management-System
# Research problem
• Limited Fire Detection: How can we provide
real-time fire alerts near critical areas like racks,
hindering rapid response 

• Inefficient Space Utilization: Lack of intelligent
systems for dynamically optimizing warehouse
space leads to inefficiencies. How can we
address that 

• Route Inefficiency: How can we save time and
energy by minimizing warehouse route
problems 

• Stock Instabilities: How to predict incidents
which address sudden movements in stock,
leading to undetected inventory issues ?
# Technologies and dependencies

Python

Matplotlib

OR-Tools

YOLOv8

PyTorch

OpenCV

AWS Lambda

Google OR

PostgreSQL
# Individual contribution

### IT21822094 | P.A.S.Tharana (Fire detection using computer vision and AI)

## Description
The Fire Detection System leverages AI-driven computer vision to detect fires in real-time, assess their severity based on proximity to critical areas (such as racks or flammable materials), and provide immediate alerts. The system integrates with a warehouse environment to enhance fire safety by enabling quicker responses and minimizing potential damage.

## Features
Real-time fire detection using computer vision.

Severity assessment based on the fire's proximity to critical zones.

Instant alerts to safety personnel regarding fire location and severity.

Integration with warehouse layouts to highlight the affected areas.

AI-powered analysis for quick and accurate fire detection

## Technologies and Dependencies
#### Frontend
OpenCV: Used for real-time image processing and visualization of fire detection.

Matplotlib: For visualizing detected fire locations and severity levels.

#### Backend
Python: Core language for AI/ML processing, model training, and running real-time fire detection algorithms.

PyTorch: Framework for training deep learning models for fire detection using computer vision.

YOLOv8: Deep learning model for fire detection and object identification in images.

#### Services
Google Cloud Vision API: Used for real-time image recognition and feature extraction.

AWS Lambda: Cloud service for scalable processing and managing real-time alerts.

Twilio: For sending real-time notifications (SMS or email) to safety personnel upon fire detection.

#### Libraries
NumPy: Efficient numerical operations for image data processing.

Pandas: Data manipulation and analysis for handling historical data or logs related to fire incidents.

Torch: AI model deployment for fire detection and severity classification.

Albumentations: Advanced data augmentation library for enhancing image data used in model training.

Flask: Used for API management to integrate the backend and trigger real-time alerts.
