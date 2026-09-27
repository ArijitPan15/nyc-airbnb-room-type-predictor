# nyc-airbnb-room-type-predictor
# NYC Airbnb Room Type Predictor

An end-to-end **Machine Learning classification web application** that predicts the room type of a New York City Airbnb listing from listing, location, availability, pricing, and review-related features.

The application uses a **Random Forest Classifier** trained on the NYC Airbnb Open Data dataset and exposes the trained model through a **FastAPI REST API**. A responsive frontend built with **HTML, CSS, and vanilla JavaScript** collects the input and displays the predicted room type and prediction probabilities.

## Project Overview

The project solves a multiclass classification problem where the target variable is `room_type`:

- `Entire home/apt`
- `Private room`
- `Shared room`

The complete workflow covered in the notebook includes:

1. Dataset loading
2. Exploratory Data Analysis (EDA)
3. Data cleaning and feature preparation
4. Train/test split
5. Feature preprocessing with `ColumnTransformer` and `Pipeline`
6. Model comparison
7. Hyperparameter tuning with `RandomizedSearchCV`
8. Final model evaluation
9. Feature importance / model analysis
10. Saving the trained preprocessing + model pipeline
11. Serving predictions through FastAPI
12. Consuming the API from the frontend

## Tech Stack

### Machine Learning

- Python 3.12.7
- Pandas
- Scikit-learn
- Joblib
- Jupyter Notebook
- Kaggle dataset

### Backend / API

- FastAPI
- Pydantic
- Uvicorn
- Pandas
- Joblib
- FastAPI CORS middleware

### Frontend

- HTML5
- CSS3
- JavaScript (Vanilla JS)

## Dataset

The model is trained using the **New York City Airbnb Open Data** dataset from Kaggle:

https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data

The notebook downloads the dataset with `kagglehub` and reads `AB_NYC_2019.csv`.

The target column is:

```text
room_type
```

The prediction features used by the API are:

```text
latitude
longitude
price
minimum_nights
number_of_reviews
reviews_per_month
calculated_host_listings_count
availability_365
neighbourhood_group
neighbourhood
```

## Machine Learning Models

The notebook compares four classification algorithms:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

The project uses **Random Forest** as the final classifier. Class balancing is enabled because the room-type classes are imbalanced.

### Cross-Validation Comparison

The notebook reports the following 3-fold cross-validation results on the training data:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression | 0.659 | 0.522 |
| Decision Tree | 0.782 | 0.647 |
| Random Forest | 0.851 | 0.715 |
| Gradient Boosting | 0.850 | 0.705 |

### Hyperparameter Tuning

`RandomizedSearchCV` is used with macro F1 as the optimization metric.

Best reported parameters:

```text
n_estimators = 200
min_samples_split = 10
max_depth = None
```

Best reported cross-validation Macro F1:

```text
0.72998
```

### Final Test Performance

The final tuned pipeline is evaluated on the untouched test set.

```text
Accuracy  : 0.85591  (~85.59%)
Macro F1  : 0.74104  (~74.10%)
```

The train/test split uses:

```text
test_size = 0.33
random_state = 42
stratify = y
```

## Project Structure

```text
NYC-Airbnb-Room-Type-Predictor-main/
│
├── Model_Pipeline.pkl
├── main.py
├── index.html
├── script.js
├── style.css
├── nyc_airbnb_room_type_classification.ipynb
├── requirements.txt
├── runtime.txt
├── the_build_line_guide.html
└── __pycache__/
```

### Important Files

**`nyc_airbnb_room_type_classification.ipynb`**  
Contains the full machine learning workflow: EDA, preprocessing, model comparison, hyperparameter tuning, evaluation, and model export.

**`Model_Pipeline.pkl`**  
Saved scikit-learn pipeline containing the preprocessing steps and the trained Random Forest model.

**`main.py`**  
FastAPI application that loads the saved model and provides the prediction endpoint.

**`index.html`**  
Frontend layout and Airbnb listing input form.

**`script.js`**  
Collects form data, calls the FastAPI prediction API, and renders the prediction and probabilities.

**`style.css`**  
Frontend styling and visual presentation.

**`requirements.txt`**  
Pinned Python packages required by the deployed API application.

**`runtime.txt`**  
Specifies the Python runtime version used for deployment.

## How the Application Works

```text
User enters Airbnb listing details
            ↓
       HTML Form
            ↓
     JavaScript / fetch()
            ↓
      POST /predict
            ↓
        FastAPI
            ↓
     Pydantic validation
            ↓
     Pandas DataFrame
            ↓
   Model_Pipeline.pkl
            ↓
   Random Forest prediction
            ↓
 Prediction + probabilities
            ↓
     JavaScript UI
            ↓
Displayed room type + probabilities
```

## API

### Health / Root Endpoint

```http
GET /
```

The current implementation returns:

```text
Hello Guyss
```

### Prediction Endpoint

```http
POST /predict
```

Example request:

```json
{
  "latitude": 40.7128,
  "longitude": -74.0060,
  "price": 150,
  "minimum_nights": 3,
  "number_of_reviews": 24,
  "reviews_per_month": 1.2,
  "calculated_host_listings_count": 1,
  "availability_365": 180,
  "neighbourhood_group": "Manhattan",
  "neighbourhood": "Chelsea"
}
```

Example response structure:

```json
{
  "Predicted_room_type": "Entire home/apt",
  "Probability": [0.91, 0.08, 0.01]
}
```

The exact probability values depend on the supplied input.

## Running Locally

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd NYC-Airbnb-Room-Type-Predictor-main
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI server

```bash
uvicorn main:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

FastAPI interactive documentation:

```text
http://127.0.0.1:8000/docs
```

## Frontend Configuration

The current `script.js` contains a configured API base URL:

```javascript
const API_BASE_URL = "https://nyc-airbnb-room-type-predictor.onrender.com";
```

For local development, change it to your local FastAPI server when required:

```javascript
const API_BASE_URL = "http://127.0.0.1:8000";
```

Then open `index.html` in a browser or serve the frontend through a local static server.

## Model Pipeline

The saved pipeline handles preprocessing and prediction together. The backend sends the validated input into the loaded pipeline rather than manually reproducing the training transformations.

The notebook uses a preprocessing pipeline based on `ColumnTransformer` and `Pipeline`, including numerical and categorical preprocessing before the Random Forest classifier.

This approach helps keep the transformations used during inference consistent with those used during training.

## Input Validation

The FastAPI/Pydantic schema validates values before prediction. Examples include:

- Latitude: `-90` to `90`
- Longitude: `-180` to `180`
- Price: must be greater than `0`
- Minimum nights: `1` to `365`
- Availability: `0` to `365`
- Review counts: non-negative
- Neighbourhood fields: required non-empty strings

## Deployment

The frontend is configured to call a deployed FastAPI service at:

```text
https://nyc-airbnb-room-type-predictor.onrender.com
```

The project includes `runtime.txt` with:

```text
python-3.12.7
```

When deploying, make sure the hosting environment uses a compatible Python and scikit-learn version because `Model_Pipeline.pkl` is a serialized scikit-learn artifact.

## Future Improvements

- Add a dedicated `/health` endpoint with a proper JSON response
- Replace `allow_origins=["*"]` with an explicit frontend origin in production
- Add automated tests for the API and input validation
- Add structured logging and error handling
- Add model versioning and experiment tracking
- Add model monitoring for prediction drift and data drift
- Improve probability calibration if probability estimates are used for decision-making
- Add Docker-based deployment
- Add CI/CD for automated testing and deployment
- Add authentication/rate limiting if the API is exposed publicly

## Learning Outcomes

This project demonstrates practical experience with:

- Supervised Machine Learning classification
- Exploratory Data Analysis
- Feature preprocessing
- Handling categorical and numerical variables
- Imbalanced classification
- Cross-validation
- Hyperparameter tuning
- Model evaluation using accuracy and Macro F1
- Model serialization with Joblib
- REST API development with FastAPI
- Pydantic request validation
- Frontend-to-backend API integration using `fetch()`
- Basic ML model deployment concepts

## Author

**Arijit Pan**  
B.Tech — Information Technology  
Backend / Python / Machine Learning
