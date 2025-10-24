# house_prices.py
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

print("🏠 TRAINING HOUSE PRICE PREDICTION AI...")
print("This predicts California house prices!")

housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)

print(f"🎯 AVERAGE PRICE PREDICTION ERROR: ${error*100000:,.2f}")
print("💡 Note: Prices are in $100,000 units")
print(f"📊 Sample prediction vs actual:")
for i in range(5):
    print(f"  Predicted: ${predictions[i]*100000:,.0f} | Actual: ${y_test[i]*100000:,.0f}")
input("Press Enter to exit...")