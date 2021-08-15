from sqlalchemy import Column, Integer, String

from database import Base


class Loan(Base):

    __tablename__ = "loan"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String)
    gender = Column(Integer)
    married = Column(Integer)
    dependents = Column(Integer)
    education = Column(Integer)
    self_employed = Column(Integer)
    applicant_income = Column(Integer)
    coapplicant_income = Column(Integer)
    loan_amount = Column(Integer)
    loan_amount_term = Column(Integer)
    credit_history = Column(Integer)
    property_area = Column(Integer)
