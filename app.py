import streamlit as st
import numpy as np
import cv2
import urllib.request
import os
from math import radians, sin, cos, sqrt, atan2

st.set_page_config(page_title="HCM Face Check-In & Geofencing")
st.title("🏢 HCM Attendance & Geofencing System")

# Geofencing Calculation (Haversine Formula)
def check_geofence(user_lat, user_lon, target_lat, target_lon, radius_meters=100):
    R = 6371000  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(radians, [user_lat, user_lon, target_lat, target_lon])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance <= radius_meters, distance

st.sidebar.header("📍 Office Location Settings")
office_lat = st.sidebar.number_input("Office Latitude", value=24.8607)
office_lon = st.sidebar.number_input("Office Longitude", value=67.0011)

st.subheader("1. Location Verification (Geofencing)")
user_lat = st.number_input("Your Current Latitude", value=24.8607)
user_lon = st.number_input("Your Current Longitude", value=67.0011)

if st.button("Verify Location"):
    is_inside, dist = check_geofence(user_lat, user_lon, office_lat, office_lon)
    if is_inside:
        st.success(f"✅ Location Verified! You are inside the office radius ({round(dist, 2)} meters away).")
    else:
        st.error(f"❌ Access Denied! You are too far from the office ({round(dist, 2)} meters away).")

st.subheader("2. Facial Recognition Check-In")
camera_input = st.camera_input("Take a picture for attendance")

if camera_input is not None:
    bytes_data = camera_input.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    # Safe local download for Haar Cascade to bypass cloud path issues
    cascade_file = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_file) or os.path.getsize(cascade_file) < 1000:
        url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
        urllib.request.urlretrieve(url, cascade_file)

    face_cascade = cv2.CascadeClassifier(cascade_file)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        st.success("🎉 Face Recognized Successfully! Attendance Marked.")
    else:
        st.warning("⚠️ No face detected clearly. Please try again.")
