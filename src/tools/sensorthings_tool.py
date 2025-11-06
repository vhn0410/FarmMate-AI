from typing import List, Dict, Any
from langchain_core.tools import tool
import os, requests, json, itertools
# --------------------------
# Configuration
# --------------------------
SENSORTHINGS_BASE = os.getenv("SENSORTHINGS_BASE", "http://localhost:8026")
OBS_SERVICE = os.getenv("OBS_SERVICE", "http://localhost:8089/ctu/geo/observations/dataStreamIds/latest")
SENSOR_USER_ID = os.getenv("SENSOR_USER_ID", "a2ecd084-3013-4a15-836c-9c0b7be0b320")


# --------------------------
# Validation for sensor values
# --------------------------

def validate_sensor_values(things: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clamp or flag obviously invalid sensor readings (e.g., pH > 14, negative EC when impossible).
    We don't 'fix' values — we annotate them so the LLM sees warnings.
    """
    for t in things:
        for sensor in t.get("sensors", []):
            for ds in sensor.get("datastreams", []):
                obs = ds.get("latest_observation")
                if not obs:
                    continue
                value = obs.get("result")
                # try to handle numeric results
                try:
                    val = float(value)
                    # pH heuristics: if unit contains 'pH' or observedProperty contains 'pH'
                    op = ds.get("observedProperty", {}) or {}
                    op_name = op.get("name", "").lower() if isinstance(op, dict) else ""
                    if "ph" in op_name:
                        if val < 0 or val > 14:
                            obs["_validator"] = f"suspicious_pH_value:{val}"
                    # temperature heuristics
                    if "temp" in op_name or "temperature" in op_name:
                        if val < -50 or val > 70:
                            obs["_validator"] = f"suspicious_temperature:{val}"
                    # EC heuristics (just example)
                    if "ec" in op_name and val < 0:
                        obs["_validator"] = f"negative_ec:{val}"
                except Exception:
                    # not numeric — skip
                    continue
    return things

# --------------------------
# Tools
# --------------------------
@tool
def sensorthings_search(query: str) -> Dict[str, Any]:
    """Return a concise JSON summary of 'things' and their latest observations.
    The tool will not dump massive raw JSON; it returns a list of devices with: id, name, sensor-count, datastream-summary.
    """
    try:
        url_thing = (
            f"{SENSORTHINGS_BASE}/get-things?"
            f"filter=properties/user_id%20eq%20%27{SENSOR_USER_ID}%27"
            "&expand=sensors($expand=datastreams($expand=observedproperties,measurementunits))"
        )
        TOKEN = os.getenv("SENSORTHINGS_TOKEN", "")
        headers = {"token": TOKEN, "Content-Type": "application/json"}
        r = requests.get(url_thing, headers=headers, timeout=10)
        if r.status_code != 200:
            return {"error": True, "message": f"Things API returned {r.status_code}", "data": None}
        things = r.json()
        # flatten
        things = list(itertools.chain.from_iterable(things))
        # get datastream ids
        datastream_ids = []
        for t in things:
            for s in t.get("sensors", []):
                for ds in s.get("datastreams", []):
                    datastream_ids.append(str(ds.get("id")))
        # observations
        obs_payload = json.dumps(datastream_ids)
        r2 = requests.post(OBS_SERVICE, headers={"accept": "application/json", "Content-Type": "application/json"}, data=obs_payload, timeout=10)
        if r2.status_code != 200:
            return {"error": True, "message": f"Observations API returned {r2.status_code}", "data": None}
        observations = r2.json()
        obs_map = {item["dataStreamId"]: item for item in observations}
        for t in things:
            for s in t.get("sensors", []):
                for ds in s.get("datastreams", []):
                    ds_id = str(ds.get("id"))
                    ds["latest_observation"] = obs_map.get(ds_id)
        # validate and annotate suspicious values
        things = validate_sensor_values(things)
        # build condensed summary to keep LLM tractable
        summary = []
        for t in things:
            t_summary = {"id": t.get("id"), "name": t.get("name"), "sensors": []}
            for s in t.get("sensors", []):
                s_summary = {"id": s.get("id"), "name": s.get("name"), "datastreams": []}
                for ds in s.get("datastreams", []):
                    ds_summary = {
                        "id": ds.get("id"),
                        "name": ds.get("name"),
                        "observedProperty": ds.get("observedProperty"),
                        "unit": ds.get("measurementUnit"),
                        "latest_observation": ds.get("latest_observation") and {
                            "result": ds.get("latest_observation").get("result"),
                            "phenomenonTime": ds.get("latest_observation").get("phenomenonTime"),
                            "_validator": ds.get("latest_observation", {}).get("_validator")
                        }
                    }
                    s_summary["datastreams"].append(ds_summary)
                t_summary["sensors"].append(s_summary)
            summary.append(t_summary)
        return {"error": False, "message": "success", "data": summary}
    except Exception as e:
        return {"error": True, "message": str(e), "data": None}