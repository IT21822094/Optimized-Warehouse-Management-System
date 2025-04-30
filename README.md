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
leading to undetected inventory issues 

# System Diagram

![Blank diagram (4)](https://github.com/user-attachments/assets/e8b816f9-91e1-474c-855d-6dc10689d2c2)



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

## Component Diagram

![Blank board (3)](https://github.com/user-attachments/assets/edd2561e-a69e-4b33-82ed-285f584a483c)


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
