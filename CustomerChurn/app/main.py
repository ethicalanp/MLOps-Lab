from fastapi import FastAPI ,HTTPException
import joblib
import pandas as pd 
from app.schema import PredictionRequest,PredictionResponse
import logging 
import time 
from prometheus_client import Counter,Histogram,generate_latest
from fastapi.responses import Response



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

#Prometheus metrics 
request_total=Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

prediction_error_total=Counter(
    "prediction_error_total",
    "Total number of prediction errors"
)

prediction_lantency_seconds = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds"
)
#Fastapi 
app=FastAPI(
    title="Customer Churn Prediction",
    description="API for predicting customer churn",
    version="1.0.0"
)

#load trained ML Pipeline 
MODEL_PATH="model/churn_pipeline.pkl"

model=joblib.load(MODEL_PATH)

logger.info("Churn model loaded successfully")

#Health Check endpoint

@app.get("/")
def home():
    return {"message":"Customer Churn Prediction API is running"}


@app.get("/health")
def health():
    return {"status":"healthy"}


@app.get("/prometheus")
def prometheus_metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )


#prediction endpoints
@app.post("/predict",response_model=PredictionResponse)
def predict(request:PredictionRequest):
    request_total.inc()

    start_time=time.perf_counter()
    logger.info("Prediction request received ")
    try:
        #validate request into dictionary
        input_data=request.model_dump()
        input_df=pd.DataFrame([input_data])

        prediction=model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0][1]

        latency = time.perf_counter() - start_time

        prediction_lantency_seconds.observe(latency)

        logger.info(
            f"Prediction complete |"
            f"Prediction={prediction} |"
            f"latency={latency :.4f}s"
        )

        return {
            "prediction": int(prediction),
            "churn":"NO" if prediction == 0 else "Yes",
            "probability":float(probability)
        }
    except Exception:
        prediction_error_total.inc()

        logger.exception(f"Prediction Failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.Please try again later"
        )

