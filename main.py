import asyncio
import pickle
import time
import uuid
from datetime import datetime

import uvicorn
from fastapi import Depends, FastAPI, Form, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

import database
import log
import models
from database import engine

app = FastAPI()

logger = log._init_logger()

models.Base.metadata.create_all(engine)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.middleware("http")
async def request_middleware(request, call_next):
    end_point = request.url.path
    request_id = str(uuid.uuid4())
    with logger.contextualize(request_id=request_id, end_point=end_point):
        logger.debug("--------------start--------------")

        try:
            return await call_next(request)
        except Exception as ex:
            print(ex)
            logger.error(f"Request failed: {ex}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": {
                        "type": "UnknownError",
                        "message": "Unknown error found. Try with different image.",
                        "fields": None,
                    },
                    "status": "failed",
                },
            )
        finally:
            logger.debug("---------------end---------------")


@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=1000)
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={
                "error": {"type": "TimeoutError", "message": "API timed out.", "fields": None},
                "status": "failed",
            },
        )


ml_model = pickle.load(open("logistic_model.pkl", "rb"))


@app.post("/loan/application/form", response_class=HTMLResponse)
async def post_data(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    gender: int = Form(...),
    married: int = Form(...),
    dependents: int = Form(...),
    education: int = Form(...),
    self_employed: int = Form(...),
    applicant_income: int = Form(...),
    coapplicant_income: int = Form(...),
    loan_amount: int = Form(...),
    loan_amount_term: int = Form(...),
    credit_history: int = Form(...),
    property_area: int = Form(...),
    db: Session = Depends(database.get_db),
):

    logger.debug(
        {
            "Name": name,
            "Email": email,
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_amount_term,
            "Credit_History": credit_history,
            "Property_Area": property_area,
        }
    )

    prediction = ml_model.predict(
        [
            [
                gender,
                married,
                dependents,
                education,
                self_employed,
                applicant_income,
                coapplicant_income,
                loan_amount,
                loan_amount_term,
                credit_history,
                property_area,
            ]
        ]
    )
    pred_loan_status = prediction[0]

    data = {
        "Name": name,
        "Email": email,
        "LoanAmount": loan_amount,
        "Result": pred_loan_status,
    }

    application_data = models.Loan(
        name=name,
        email=email,
        gender=gender,
        married=married,
        dependents=dependents,
        education=education,
        self_employed=self_employed,
        applicant_income=applicant_income,
        coapplicant_income=coapplicant_income,
        loan_amount=loan_amount,
        loan_amount_term=loan_amount_term,
        credit_history=credit_history,
        property_area=property_area,
    )
    db.add(application_data)
    db.commit()
    db.refresh(application_data)

    return templates.TemplateResponse("loan.html", {"request": request, "data": data})


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("loan_application.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8001)
