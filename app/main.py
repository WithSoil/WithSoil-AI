# app/main.py
import os
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

from api.v1.router import api_router
from api.v1.endpoints.diagnose import disease_service


app = FastAPI(title="SmartFarm AI Server")

app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
def startup_event():
    disease_service.load_model()

@app.get("/")
def read_root():
    return {"message": "FastAPI AI Server is running"}