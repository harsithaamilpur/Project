from flask import Flask, jsonify, render_template, request, redirect, url_for
import csv, os, datetime, random, joblib
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import numpy as np
import requests
from models import PricingModel

app = Flask(__name__, template_folder='.')
csv_file = 'initial_dataset.csv'
feedback_file = 'ride_feedback.csv'

try:
    model = joblib.load('pricing_model.pkl') if os.path.exists('pricing_model.pkl') else None
except Exception as e:
    print(f"Model loading failed: {e}")
    model = None

geolocator = Nominatim(user_agent="mobility_app")

DRIVERS = [
    {"id": "DRV_1001", "name": "Rajesh K.", "vehicle": "Tata Nexon EV", 
     "vehicle_no": "KA01AB1234", "safe_rides": 142, "rating": 4.7, "priority_ready": True, "photo": "driver1.jpg"},
    {"id": "DRV_1002", "name": "Priyansh M.", "vehicle": "Mahindra eVerito", 
     "vehicle_no": "KA02CD5678", "safe_rides": 89, "rating": 4.5, "priority_ready": False, "photo": "driver2.jpg"},
    {"id": "DRV_1003", "name": "Abdul S.", "vehicle": "Maruti WagonR", 
     "vehicle_no": "KA03EF9012", "safe_rides": 115, "rating": 4.9, "priority_ready": True, "photo": "driver3.jpg"}
]

def init_files():
    for filepath in [csv_file, feedback_file]:
        if not os.path.exists(filepath):
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                if filepath == csv_file:
                    writer.writerow([
                        'timestamp', 'pickup', 'dropoff', 'distance_km', 
                        'priority', 'carpool', 'driver_id', 'final_price',
                        'route_geometry', 'distance_squared', 'priority_distance_interaction'
                    ])
                else:
                    writer.writerow([
                        'timestamp', 'driver_id', 'ride_rating', 'route_efficiency',
                        'carpool_interest', 'traffic_avoidance', 'recommendation'
                    ])

def get_route(pickup_loc, dropoff_loc):
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{pickup_loc[1]},{pickup_loc[0]};{dropoff_loc[1]},{dropoff_loc[0]}?overview=full&geometries=geojson"
        response = requests.get(url, timeout=5).json()
        return response['routes'][0]['geometry']['coordinates']
    except Exception as e:
        print(f"Routing error: {e}")
        return None

def calculate_price(distance, priority=False, carpool=False):
    try:
        base_price = distance * 10
        if priority and carpool:
            final_price = base_price * 0.90
        elif priority:
            final_price = base_price * 1.20
        elif carpool:
            final_price = base_price * 0.80
        else:
            final_price = base_price
            
        if model:
            input_features = np.array([[
                distance,
                int(priority),
                int(carpool),
                distance ** 2,
                int(priority) * distance
            ]])
            model_adjustment = model.predict(input_features)[0]
            final_price = (final_price + model_adjustment) / 2
            
        return round(max(final_price, distance * 5), 2)
    except Exception as e:
        print(f"Pricing error: {e}")
        return round(distance * 10, 2)

@app.route('/favicon.ico')
def favicon():
    return '', 404

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/find_ride', methods=['POST'])
def find_ride():
    try:
        pickup = request.form.get('pickup', '').strip()
        dropoff = request.form.get('dropoff', '').strip()
        if not pickup or not dropoff:
            return redirect(url_for('index'))
            
        priority = request.form.get('priority') == 'on'
        carpool = request.form.get('carpool') == 'on'
        
        try:
            pickup_loc = geolocator.geocode(pickup, timeout=10)
            dropoff_loc = geolocator.geocode(dropoff, timeout=10)
            pickup_coords = (pickup_loc.latitude, pickup_loc.longitude)
            dropoff_coords = (dropoff_loc.latitude, dropoff_loc.longitude)
            distance = round(geodesic(pickup_coords, dropoff_coords).km, 2)
            route = get_route(pickup_coords, dropoff_coords)
        except Exception as e:
            print(f"Geocoding fallback: {e}")
            distance = round(random.uniform(5, 15), 2)
            route = None
        
        price = calculate_price(distance, priority, carpool)
        eligible_drivers = [d for d in DRIVERS if not priority or d['priority_ready']]
        driver = random.choice(eligible_drivers) if eligible_drivers else DRIVERS[0]
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.datetime.now().isoformat(),
                pickup, dropoff, distance,
                priority, carpool, driver['id'], price,
                str(route) if route else '',
                distance ** 2,
                int(priority) * distance
            ])
        
        return render_template('result.html',  
            pickup=pickup,
            dropoff=dropoff,
            distance=distance,
            price=price,
            driver=driver,
            priority=priority,
            carpool=carpool,
            route_coords=route
        )
    except Exception as e:
        print(f"Ride error: {e}")
        return redirect(url_for('index'))

@app.route('/show_feedback_form', methods=['POST'])
def show_feedback_form():
    try:
        driver_id = request.form.get('driver_id')
        if not driver_id:
            return redirect(url_for('index'))
        return render_template('feedback.html', driver_id=driver_id)
    except Exception as e:
        print(f"Feedback form error: {e}")
        return redirect(url_for('index'))

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        with open(feedback_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.datetime.now().isoformat(),
                request.form.get('driver_id', ''),
                request.form.get('rating', ''),
                request.form.get('efficiency', ''),
                request.form.get('carpool', ''),
                request.form.get('traffic', ''),
                request.form.get('recommend', '')
            ])
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Feedback Submitted</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <meta http-equiv="refresh" content="3;url=/">
        </head>
        <body>
            <div class="container" style="max-width: 600px; margin-top: 50px;">
                <div class="alert alert-success text-center">
                    <h4>Feedback successfully submitted!</h4>
                    <p>Redirecting to homepage in 3 seconds...</p>
                </div>
            </div>
        </body>
        </html>
        """
    except Exception as e:
        print(f"Feedback error: {e}")
        return redirect(url_for('index'))

if __name__ == '__main__':
    init_files()
    try:
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"Failed to start server: {e}")
        input("Press Enter to exit...")