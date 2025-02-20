# Machine Learning Based Enhanced Road Safety

A comprehensive project integrating **drowsiness detection**, **automated license plate recognition**, and **ticketing analysis** to enhance **road safety** and **traffic rule enforcement**. This repository demonstrates a full end-to-end **Data Science pipeline** using multiple datasets, machine learning (ML) models, and a web-based dashboard application.

---

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Project Structure](#project-structure)
4. [Datasets](#datasets)
5. [Methodology](#methodology)
6. [Installation and Setup](#installation-and-setup)
7. [Running the Application](#running-the-application)
8. [Results and Evaluation](#results-and-evaluation)
9. [Lessons Learned](#lessons-learned)
10. [Future Scope](#future-scope)
11. [References](#references)

---

## Overview

Modern urban transportation faces critical challenges in **road safety**, **traffic rule enforcement**, and **parking management**. This project addresses these issues by leveraging **Machine Learning** and **Data Analysis** to develop solutions for:
1. **Driver Drowsiness Detection** using a CNN-based model.
2. **License Plate Recognition** (LPR) using a fine-tuned YOLO model.
3. **Ticketing Analysis** through a dashboard highlighting trends in parking and traffic infractions.

Our end-product is an interactive **Flask**-based web application providing:
- Real-time driver drowsiness status (Yawning/Not Yawning, Eyes Open/Closed).
- Real-time license plate recognition.
- Visualization of ticketing patterns and trends in **Toronto** (though it can be extended to other regions).

---

## Key Features

1. **GeoJson + Parking Tickets Analysis**  
   - Spatial merging of **GeoJson** boundaries with **parking tickets** data to map infractions by neighborhood.

2. **Driver Drowsiness Detection**  
   - **Convolutional Neural Network (CNN)** classifies driver states: eyes open/closed and yawn/no yawn.
   - Real-time or batch image inference through the web interface.

3. **License Plate Recognition**  
   - Fine-tuned **YOLO** model trained on license plate images.
   - Automatically detects plates and extracts text (via [Pytesseract](https://github.com/madmaze/pytesseract) or integrated OCR).

4. **Comprehensive Ticketing Analysis**  
   - Interactive data visualizations (using **Matplotlib** and **Plotly**) for ticket distribution by time, location, type, and amount.
   - Heatmaps, bar charts, line plots to identify parking “hotspots” and potential enforcement optimizations.

5. **Web Application Dashboard**  
   - Built with **Flask** (Python backend) + **HTML/CSS/Bootstrap** (frontend) + **JavaScript** (interactivity).
   - Three primary tabs:
     1. **Visualizations**: Explore ticketing data through interactive charts.
     2. **License Plate Recognition**: Upload images with vehicles to detect & read license plates.
     3. **Driver Drowsiness**: Upload driver images to detect drowsiness status.

---

## Project Structure

A suggested directory layout:

```
.
├── data
│   ├── drowsiness_dataset/       # Images for drowsiness detection
│   ├── license_plate_dataset/    # Images/annotations for LPR
│   ├── parking_tickets/          # Parking tickets data
│   ├── driver_stats/             # PDF or CSV data for driver demographics
│   └── geojson/                  # Toronto neighborhoods GeoJson
│
├── notebooks
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_drowsiness_detection.ipynb
│   └── 04_license_plate_recognition.ipynb
│
├── models
│   ├── yolo_model/               # YOLO weights, config
│   └── cnn_model.h5              # Trained CNN for drowsiness
│
├── static
│   ├── css/                      # CSS files
│   └── js/                       # JavaScript files
│
├── templates
│   ├── base.html
│   ├── index.html
│   ├── license_plate.html
│   └── drowsiness.html
│
├── app.py                        # Flask app entry point
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## Datasets

1. **Drowsiness Dataset**  
   - **Link**: [Kaggle - Drowsiness Dataset](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset)  
   - **Description**: Images of drivers covering four classes: closed eyes, open eyes, yawn, no yawn.

2. **Parking Tickets Dataset**  
   - **Link**: [Toronto Open Data - Parking Tickets](https://open.toronto.ca/dataset/parking-tickets/)  
   - **Description**: Contains non-identifiable parking ticket records for each calendar year in Toronto.

3. **Toronto Neighborhoods Dataset**  
   - **Link**: [Toronto Neighbourhoods GeoJSON](https://open.toronto.ca/dataset/neighbourhoods/)  
   - **Description**: GeoJSON file with boundary coordinates for each neighborhood in Toronto.

4. **Driver Population Statistics**  
   - **Link**: [Ontario Data - Driver Population Statistics](https://data.ontario.ca/en/dataset/driver-population-statistics)  
   - **Description**: Annual statistics on driver demographics and licensing.

5. **License Plates Dataset**  
   - **Link**: [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html#download-manually)  
   - **Description**: Images containing vehicle license plates, used for bounding box annotation.

---

## Methodology

1. **Data Gathering & Integration**  
   - Merged the **GeoJson** boundary data with **parking ticket** data via spatial joins (using [Shapely](https://github.com/shapely/shapely)) to attribute tickets to neighborhoods.

2. **Data Cleaning, Processing, and Visualization**  
   - Used **Python**, **Pandas**, **NumPy** to clean and preprocess data.
   - **Matplotlib** and **Plotly** for exploratory data analysis (EDA) and visualizations.

3. **Driver Population Statistics**  
   - Scraped PDF tables using **Tabula** or other tools for demographics; integrated with ticket data to discover potential correlations.

4. **License Plate Recognition**  
   - Annotated images using **labelImg** to create bounding boxes around plates.
   - Fine-tuned a **YOLO** model (Darknet-based) for accurate, real-time license plate detection.
   - Used **OCR** (Pytesseract or integrated approach) to extract text from detected plates.

5. **Driver Drowsiness Detection**  
   - Employed a **CNN** to classify four driver states (open/closed eyes, yawn/no yawn).
   - Model architecture includes multiple **Conv2D** + **MaxPooling** layers, plus **Dropout** to reduce overfitting.
   - Achieved high accuracy (~96%+ on validation).

6. **Web Application Development**  
   - **Flask** backend routes for each feature:
     1. **Ticketing Analysis**: Renders plots, interactive graphs.
     2. **License Plate**: Accepts image uploads, calls YOLO + OCR, displays recognized text.
     3. **Drowsiness**: Accepts driver images, runs CNN inference, displays classification results.
   - **HTML/CSS/Bootstrap** for responsive UI; **JavaScript** for interactivity.

---

## Installation and Setup

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/<your_username>/<repo_name>.git
   cd <repo_name>
   ```

2. **Create and Activate a Virtual Environment (optional but recommended)**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Mac/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
   Make sure you have **Python 3.7+** installed.

4. **Download/Place Datasets**  
   - Place **drowsiness**, **license plate**, **parking ticket**, **driver stats**, and **GeoJson** files in the `data` folder as per the [Datasets](#datasets) section.

5. **Configure YOLO Model (for License Plate Recognition)**  
   - Download pretrained YOLO weights (e.g., YOLOv5 or YOLOv3/v4 Darknet weights).
   - Place them under the `models/yolo_model/` folder.
   - Adjust config or `.yaml` files accordingly to point to your dataset classes.

---

## Running the Application

1. **Start the Flask App**  
   ```bash
   python app.py
   ```
2. **Open Your Browser**  
   - Navigate to `http://127.0.0.1:5000/` or the displayed local server URL.

3. **Explore Features**  
   - **Ticketing Analysis**: Interactive graphs, heatmaps, distribution plots.
   - **License Plate Recognition**: Upload a vehicle image to detect and read license plates.
   - **Driver Drowsiness**: Upload a driver image to classify drowsiness state.

---

## Results and Evaluation

### 1. Ticketing Analysis

- **Distribution of Ticket Amounts**: Found the majority of tickets are for the lowest fine, with only a small fraction at higher fines.  
- **Time-Based Trends**: Identified seasonal peaks (e.g., March to October) and daily patterns (weekdays vs. weekends).  
- **Neighborhood Hotspots**: Heatmaps reveal specific areas in Toronto with the highest density of parking infractions.

### 2. License Plate Recognition

- **YOLO Precision**: ~1.00 (highly accurate predictions).  
- **Recall**: ~0.958 (model rarely misses plates).  
- **mAP (IoU 0.5)**: ~0.993, indicating robust bounding box predictions.  
- **OCR**: Successfully extracts plate numbers except in cases of severe blurriness or occlusion.

### 3. Driver Drowsiness Detection

- **CNN Architecture**:  
  - Conv2D layers: 256, 128, 64, 32 filters  
  - Dropout: 0.5  
  - Final Dense layers for classification (4 classes)  
- **Accuracy**: ~0.9689 on validation data.  
- **Loss**: Converged steadily, minimal gap between training and validation sets, indicating good generalization.

---

## Lessons Learned

1. **Data Integration & Cleaning**  
   - Combined disparate data sources (GeoJson, PDF scraping, CSV files) using advanced library support (Pandas, Shapely).

2. **Choosing Pre-trained Models**  
   - Speed and accuracy benefits from using **YOLO** vs. writing an object detection model from scratch.

3. **End-to-End Development**  
   - Building a full-stack solution (Frontend + Backend + ML) taught the importance of seamless integration and optimization.

4. **Project Management**  
   - Adhering to timelines, iterative testing, contingency planning for inevitable bugs and performance issues.

---

## Future Scope

- **Weather Conditions Prediction**  
  - Integrate weather data for real-time alerts (slippery roads, fog warnings, etc.).

- **Pedestrian Interaction Analysis**  
  - Detect interactions between vehicles and pedestrians using advanced tracking and object detection.

- **Emergency Response Integration**  
  - Provide automatic alerts to emergency services in case of accidents or critical drowsiness detection.

These future enhancements will further **improve safety**, **reduce accidents**, and **optimize traffic enforcement** using data-driven insights.

---

## References

1. [Face Detection Algorithm for Drowsiness Detection](https://ieeexplore.ieee.org/document/9182237)  
2. [Drowsiness Detection via CNN](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7917813/)  
3. [Automated License Plate Recognition Systems](https://ieeexplore.ieee.org/abstract/document/8343528)  
4. [YOLO-based License Plate Detection Enhancement](https://ieeexplore.ieee.org/document/10071305)

---

**Video Demo**: [Watch Project Demo](https://youtu.be/wa_gxY8OSuE)
