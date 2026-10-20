import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

print("==================================================")
print(" SIMULASI HPC & AI UNTUK PREDIKSI KUALITAS UDARA (PM2.5)")
print(" Studi Kasus: Mitigasi Tantangan Lingkungan Lokal")
print("==================================================\n")

def generate_hpc_environmental_data(n_samples=5000):
    np.random.seed(42)
    data = {
        'temperature': np.random.uniform(20.0, 38.0, size=n_samples),
        'humidity': np.random.uniform(30.0, 95.0, size=n_samples),
        'wind_speed': np.random.uniform(0.5, 15.0, size=n_samples),
        'traffic_density': np.random.randint(100, 5000, size=n_samples),
        'industrial_output': np.random.uniform(10.0, 200.0, size=n_samples)
    }
    
    pm25 = (
        2.5 * data['temperature'] - 
        0.8 * data['humidity'] - 
        1.2 * data['wind_speed'] + 
        0.015 * data['traffic_density'] + 
        0.4 * data['industrial_output'] + 
        np.random.normal(0, 5.0, size=n_samples)
    )
    data['pm25_level'] = np.clip(pm25, 0, 500)
    return pd.DataFrame(data)

print("[INFO] Memproses data sensor lingkungan secara paralel...")
start_time = time.time()
df = generate_hpc_environmental_data(10000)
print(f"[INFO] Berhasil memproses {len(df)} baris data dalam {time.time() - start_time:.4f} detik.\n")

X = df[['temperature', 'humidity', 'wind_speed', 'traffic_density', 'industrial_output']]
y = df['pm25_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("[INFO] Melatih model Machine Learning berbasis Ensemble...")
model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1) 
model.fit(X_train, y_train)
print("[INFO] Pelatihan model selesai!\n")

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("==================================================")
print(" EVALUASI KINERJA MODEL PREDIKSI")
print("==================================================")
print(f"Mean Squared Error (MSE) : {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared Score (R2)     : {r2:.4f}")
print("==================================================")