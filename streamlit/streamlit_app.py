import requests
import streamlit as st

PROMETHEUS_URL = "http://prometheus:9090"  # inside docker network
# if run streamlit on host (not in docker): use http://localhost:9090 PILAS

st.set_page_config(page_title="Inference Metrics", layout="wide")
st.title("Inference API – Live Metrics")

def prom_query(q: str):
    resp = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": q})
    resp.raise_for_status()
    data = resp.json()
    if data["status"] != "success":
        return None
    return data["data"]["result"]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Total requests")
    res = prom_query("predict_requests_total")
    total = 0
    if res:
        # sum over all series
        total = sum(float(r["value"][1]) for r in res)
    st.metric("Requests", f"{int(total)}")

with col2:
    st.subheader("Current QPS (approx)")
    # rate over last 5m
    res = prom_query("rate(predict_requests_total[5m])")
    qps = 0.0
    if res:
        qps = sum(float(r["value"][1]) for r in res)
    st.metric("req/s", f"{qps:.2f}")

st.subheader("Latency buckets (last scrape)")
res = prom_query("predict_latency_seconds_bucket")
if res:
    for series in res:
        st.write(series["metric"], series["value"])
else:
    st.write("No latency data yet.")
 