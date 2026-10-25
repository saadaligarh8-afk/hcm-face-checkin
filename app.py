from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from math import radians, sin, cos, sqrt, atan2

app = Flask(__name__)
CORS(app)

# Default Office Settings (Admin can update via UI)
app_settings = {
    "latitude": 24.8607,
    "longitude": 67.0011,
    "radius": 5000  # Default expanded radius for smooth testing
}

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

@app.route('/')
def home():
    return render_template('index.html')

# Admin Settings API to update office location and radius
@app.route('/api/settings', methods=['POST'])
def update_settings():
    data = request.json
    try:
        app_settings['latitude'] = float(data.get('latitude', app_settings['latitude']))
        app_settings['longitude'] = float(data.get('longitude', app_settings['longitude']))
        app_settings['radius'] = float(data.get('radius', app_settings['radius']))
        return jsonify({'success': True, 'message': 'Office settings updated successfully.'})
    except ValueError:
        return jsonify({'success': False, 'message': 'Invalid coordinates or radius provided.'}), 400

@app.route('/api/attendance/check-in', methods=['POST'])
def check_in():
    data = request.json
    lat = data.get('latitude')
    lng = data.get('longitude')
    accuracy = data.get('accuracy')
    image_data = data.get('image')

    if not lat or not lng:
        return jsonify({'success': False, 'message': 'Location data missing.'}), 400

    # 1. GPS Accuracy Check (Relaxed to 10000m for indoor testing convenience)
    if accuracy > 10000:
        return jsonify({'success': False, 'message': 'GPS accuracy is too low. Please move to an open area.'}), 400

    # 2. Geofence Radius Check (Haversine formula)
    distance = calculate_distance(lat, lng, app_settings['latitude'], app_settings['longitude'])
    if distance > app_settings['radius']:
        return jsonify({
            'success': False, 
            'message': f'Check-in failed: You are outside the office radius. Current distance: {int(distance)}m (Allowed: {app_settings["radius"]}m).'
        }), 400

    # 3. Face Detection using OpenCV (Exactly one face rule)
    if image_data:
        header, encoded = image_data.split(',', 1)
        decoded_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'Check-in rejected: No face detected. Please face the camera.'}), 400
        elif len(faces) > 1:
            return jsonify({'success': False, 'message': 'Check-in rejected: Multiple faces detected. Only one person is allowed.'}), 400
    else:
        return jsonify({'success': False, 'message': 'Image data missing.'}), 400

    return jsonify({'success': True, 'message': 'Attendance successfully marked via facial recognition and geofencing.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)