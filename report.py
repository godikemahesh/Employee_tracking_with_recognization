import streamlit as st
import pandas as pd
import json
from datetime import datetime

from streamlit_autorefresh import st_autorefresh


# ---------- SETTINGS ----------
JSON_PATH = "report1.json"
REFRESH_SECONDS = 10

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Employee Tracking Dashboard",
                   layout="wide")

st.title("Employee Tracking Dashboard")
st.caption("Live CCTV Based Attendance Monitoring System")

# Auto Refresh
st_autorefresh(interval=REFRESH_SECONDS * 1000, key="datarefresh")

try:
    with open(JSON_PATH, "r") as f:
        all_data = json.load(f)

    all_dates = sorted(all_data.keys(),
                       key=lambda d: datetime.strptime(d, "%d-%m-%Y"),
                       reverse=True)
    
    import datetime
  
    all_dates = [datetime.datetime.strptime(d, "%d-%m-%Y").date() for d in all_dates]
    
    selected_date = st.date_input(
        "📅 Select Date",
        value=all_dates[0],
        min_value=min(all_dates),
        max_value=max(all_dates)
    )

    if not all_dates:
        st.warning("No report data found.")
        st.stop()


    employees = all_data.get(selected_date.strftime("%d-%m-%Y"), {})

    rows = []
    present_count = 0
    absent_count = 0
    total_breaks = 0

    def fmt(d, key_time, key_sec):
        if key_sec in d:
            secs = int(d[key_sec])
            return f"{secs//3600}h {(secs%3600)//60}m"
        if key_time in d:
            return d[key_time]
        return "0h 0m"

    for name, details in employees.items():

        state = details["current_state"]

        if state == "PRESENT":
            present_count += 1
        else:
            absent_count += 1

        total_breaks += details["total_breaks"]

        rows.append({   
            "Employee Name": name.capitalize(),
            "Current Status": state,
            "In Seat Time": fmt(details, "in_seat_time", "in_seat_seconds"),
            "Out Seat Time": fmt(details, "out_seat_time", "out_seat_seconds"),
            "Total Breaks": details["total_breaks"]
        })

    df = pd.DataFrame(rows)

    # -------- Summary Cards --------
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("👥 Total Employees", len(employees))
    col2.metric("🟢 Present", present_count)
    col3.metric("🔴 Absent", absent_count)
    #col4.metric("☕ Total Breaks", total_breaks)
    with open("time.json", "r") as f:
        col4.metric("⏰ Last updated at",f.read())
    #col5.metric("started at: ",datetime.now().strftime('%H:%M:%S'))
    st.divider()
    # st.button("Add employee zones", on_click=draw_desk.main)
    # st.button("Register employee", on_click=employee_register.main)
    # -------- Employee Table --------
    st.subheader("📋 Employee Tracking Sheet")

    st.dataframe(df,width="stretch")

except Exception as e:
    st.error(f"Error reading JSON: {e}")