from locust import FastHttpUser, task, between
import os
from datetime import datetime

# Full schema taken from diabetes_curated (minus the columns the DAG drops)
# i.e. what the training DAG actually used as X
def build_diabetes_record():
    return {
        # categoricals
        "race": "Caucasian",
        "gender": "Female",
        "age": "[70-80)",
        "weight": None,  # dataset often uses "?" or NULL; model handles via pipeline
        # IDs / ints
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        # numeric clinical counts
        "time_in_hospital": 3,
        "payer_code": None,
        "medical_specialty": "Emergency/Trauma",
        "num_lab_procedures": 41,
        "num_procedures": 0,
        "num_medications": 13,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "diag_1": "250.83",
        "diag_2": "401.9",
        "diag_3": "V58.67",
        "number_diagnoses": 6,
        "max_glu_serum": None,
        "A1Cresult": "Norm",
        # meds — these MUST be present because your DF had them
        "metformin": "No",
        "repaglinide": "No",
        "nateglinide": "No",
        "chlorpropamide": "No",
        "glimepiride": "No",
        "acetohexamide": "No",
        "glipizide": "No",
        "glyburide": "No",
        "tolbutamide": "No",
        "pioglitazone": "No",
        "rosiglitazone": "No",
        "acarbose": "No",
        "miglitol": "No",
        "troglitazone": "No",
        "tolazamide": "No",
        "examide": "No",
        "citoglipton": "No",
        "insulin": "Steady",
        "glyburide-metformin": "No",
        "glipizide-metformin": "No",
        "glimepiride-pioglitazone": "No",
        "metformin-rosiglitazone": "No",
        "metformin-pioglitazone": "No",
        "change": "Ch",
        "diabetesMed": "Yes",
        # this column stayed in curated and is NOT dropped in training
        "served_at": datetime.utcnow().isoformat(),
    }


class InferenceUser(FastHttpUser):
    wait_time = between(0.5, 2.0)
    host = os.getenv("TARGET_HOST", "http://localhost:8989")

    def on_start(self):
        # optional warmup / docs call
        self.client.get("/model/expected-schema", name="/model/expected-schema", timeout=20)

    @task
    def predict(self):
        record = build_diabetes_record()
        payload = {"records": [record]}
        with self.client.post(
            "/predict",
            json=payload,
            name="/predict",
            timeout=20,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text}")
            else:
                data = resp.json()
                if "predictions" not in data:
                    resp.failure("no 'predictions' in response")
