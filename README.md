# Optimized-Warehouse-Management-System
# Introduction


The Optimized Warehouse Management System is designed to revolutionize warehouse operations through the strategic use of artificial intelligence, computer vision, and operational best practices. It focuses on four critical areas to maximize efficiency, safety, and productivity
# Git repo link
https://github.com/IT21822094/Optimized-Warehouse-Management-System
# Research problem
• Limited Fire Detection(Vision-Based Fire Detection): How can we leverage existing cameras with 
AI to detect early-stage fires, assess risk, and predict spread in warehouses for faster response

• Inefficient Space Utilization: Lack of intelligent
systems for dynamically optimizing warehouse
space leads to inefficiencies. How can we
address that 

• Route Inefficiency: How can we save time and
energy by minimizing warehouse route
problems 

• Stock Instabilities: How to predict incidents
which address sudden movements in stock,
leading to undetected inventory issues 

# System Diagram

![Blank diagram](https://github.com/user-attachments/assets/093d69ba-1dc5-4736-92b9-aaf71060cb14)




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

### IT21822094 | P.A.S.Tharana (Vision-Based Fire Detection System Using AI and IoT)
## Description
This enhanced fire detection system integrates computer vision, AI, and IoT hardware to provide an intelligent, real-time fire monitoring solution for warehouses. Using only CCTV cameras and a low-cost IoT alert system, the system detects and monitors fires or smoke instantly.
It classifies the fire’s severity, determines its proximity to shelves or assets, predicts fire spread direction, and triggers visual and audio alerts using LEDs and buzzers. The solution is fully scalable, cost-efficient, and eliminates the need for expensive fire sensors or cameras.


## Features
Camera + IoT Integration – Combines AI-based vision detection with IoT sensors (MQ-7 smoke sensors, LEDs, buzzers) for enhanced accuracy.

Camera-Only Operation (Optional) – Works solely on CCTV input when IoT devices are unavailable.

Real-Time Fire & Smoke Detection – Detects fire or smoke instantly using deep learning models.

Fire Size Classification – Categorizes fire as small, medium, or large based on detected pixel area.

Shelf Proximity Detection – Identifies warehouse shelves and calculates the distance between fire and nearby assets.

Fire Spread Prediction – Uses motion and heat pattern analysis to estimate possible fire spread direction.

MQ-7 Smoke Sensors: Detect smoke concentration and verify fire presence.

LED Indicators: Eight LEDs arranged directionally (up, down, left, right) show where fire or smoke is detected.

Buzzers: Four buzzers trigger loud alerts corresponding to fire zones.

Instant Alerts – Sends sound, light, and mobile notifications via Twilio when a fire event is detected.

Live Dashboard – Displays real-time video feed with detected fire zones, risk levels, and IoT sensor data.

Data Logging – Records fire incidents, alerts, and sensor readings for later analysis.

Low-Cost Deployment – Uses existing CCTV cameras and low-cost IoT modules.

Scalable Design – Can monitor small storage rooms or large industrial warehouses.
## Component Diagram

<img width="2696" height="1553" alt="Blank board (1)" src="https://github.com/user-attachments/assets/469a32bf-8d55-4fac-837d-c54e93940921" />




## Technologies and Dependencies
### Frontend
OpenCV: Used for real-time image processing and visualization of fire detection.

Matplotlib: For visualizing detected fire locations and severity levels.

### Backend
Python: Core language for AI/ML processing, model training, and running real-time fire detection algorithms.

PyTorch: Framework for training deep learning models for fire detection using computer vision.

YOLOv8: Deep learning model for fire detection and object identification in images.

### Services
Google Cloud Vision API: Used for real-time image recognition and feature extraction.

AWS Lambda: Cloud service for scalable processing and managing real-time alerts.

Twilio: For sending real-time notifications (SMS or email) to safety personnel upon fire detection.

### Libraries
NumPy: Efficient numerical operations for image data processing.

Pandas: Data manipulation and analysis for handling historical data or logs related to fire incidents.

Torch: AI model deployment for fire detection and severity classification.

Albumentations: Advanced data augmentation library for enhancing image data used in model training.

Flask: Used for API management to integrate the backend and trigger real-time alerts.

### IoT and Hardware
Arduino  – Microcontroller for IoT integration.

MQ-7 Smoke Sensors – Detect smoke and verify fire presence.

LED Indicators (x8) – Indicate fire/smoke direction (up, down, left, right).

Buzzers (x4) – Emit audible alerts based on fire location.

---------------------------------------------------------------------------------------------------------------------
### IT21822780 | Abeydeera A A A S (maximize space utilization.)

## Description
The Container Space Optimization module is designed to intelligently arrange warehouse products to maximize the use of available volume (CBM). By dynamically selecting the best storage locations and visualizing product placement, it reduces space wastage, enhances accessibility, and improves operational efficiency. This component plays a vital role in minimizing operational costs and supporting smarter warehouse management through AI-based decisions.

## Features
Dynamic product arrangement based on CBM analysis.

Real-time 2D visualization of product storage layouts.

Custom best-fit algorithm to reduce unused space in racks.

Systematic placement strategy to improve accessibility and retrieval efficiency.

Intelligent vertical and horizontal space utilization analysis.

## Component Diagram

![Blank board](https://github.com/user-attachments/assets/c0f65fe3-b3e3-410b-a8d9-1cc49b3f69cb)

## Technologies and Dependencies

### Frontend
Matplotlib: Used for visualizing the optimized product placement within warehouse racks.

### Backend
Python: Core language for implementing algorithms and logic behind space optimization.
Custom Best-Fit Algorithm: Developed in Python to calculate optimal arrangement of products in constrained spaces.

### Services
architecture supports integration with warehouse layout databases.

### Libraries
NumPy: For numerical computations in volume calculations and placement algorithms.
Matplotlib: For real-time and animated 2D visualizations of the storage arrangement.
Pandas: For managing product data and CBM information during arrangement decisions.

------------------------------------------------------------------------------------------------
### IT21079672 | Palihena P.D.M.P (Order picking route optimization)

## Description
The Order Picking Route Optimization system aims to minimize the time and effort spent by workers in retrieving items across warehouse floors. By applying real-time pathfinding algorithms, the system generates the most efficient route for each picking task. It dynamically adjusts to warehouse conditions such as obstacles, ongoing operations, and congestion, improving safety, reducing operational delays, and increasing overall productivity.

## Features
Real-time route planning for order pickers and forklifts.

Dynamic path adjustment in response to live worker and vehicle positions.

Collision and congestion avoidance using GPS and obstacle data.

Visual mapping of routes over warehouse layouts.

Time and energy-efficient order fulfillment through optimal navigation paths.

## Component Diagram

![image](https://github.com/user-attachments/assets/6380de79-709e-43a7-ac68-988aa2460e1a)

## Technologies and Dependencies

### Frontend
Matplotlib: Used to visually represent optimal routes and simulate navigation paths through the warehouse.

### Backend
Python: Core language for implementing pathfinding algorithms and backend logic.
Google OR-Tools: Used to solve routing problems such as the Traveling Salesman Problem (TSP).
Customized A* Algorithm: Tailored pathfinding algorithm for real-time route optimization under constraints.

### Services
AWS Lambda: Enables scalable cloud-based route computations and updates.
Google Maps/GPS Integration (if used): For real-time location tracking of workers and forklifts.

### Libraries
NumPy: For grid-based route calculations and matrix representations of warehouse layouts.
Flask: To expose backend services and API endpoints for frontend-to-backend communication.
PostgreSQL: Stores warehouse layouts, order information, and route history for analysis.

--------------------------------------------------------------------------------------------------------
###  IT21318184 | Silva V S  (stock movement instabilities by analyzing historical inventory data.)

## Description
The Stock Movement Instability Detection system is designed to monitor and analyze inventory movement using historical data. It identifies abnormal patterns that may indicate theft, misplacement, or operational anomalies. By applying time-series forecasting and anomaly detection algorithms, it generates real-time alerts and visualizations to help warehouse staff quickly investigate and resolve issues, ensuring inventory accuracy and improving safety.

## Features
Real-time detection of abnormal stock movements.

Uses historical inventory data for predictive modeling and baseline generation.

Configurable threshold-based alerts for deviations in inbound/outbound patterns.

Visual mapping of anomalies on digital warehouse layouts.

Automated notifications to relevant personnel via multiple channels (email/SMS).

## Component Diagram
![image](https://github.com/user-attachments/assets/28bf3fb2-d893-42fe-b50e-ac1d2b658e8c)

## Technologies and Dependencies

### Frontend
Matplotlib: For plotting historical vs. forecasted trends and visualizing anomalies.
Custom dashboard (planned): Displays warehouse maps with highlighted alert zones.

### Backend
Python: Core language for data analysis, forecasting, and backend logic.
Facebook Prophet: Time-series forecasting library used to predict normal stock movement trends.
Custom Anomaly Detection Algorithms: Based on configurable percentage deviations from forecasted values.

### Services
AWS Lambda: For real-time processing of stock data and triggering alerts.
Twilio: For sending immediate alerts to warehouse staff via SMS or email.

### Libraries
Pandas: For data manipulation, aggregation, and transformation of historical records.
NumPy: Used in statistical operations and threshold computations.
Flask: API handling and communication between components.
Scikit-learn (optional): For additional anomaly detection techniques if integrated.
