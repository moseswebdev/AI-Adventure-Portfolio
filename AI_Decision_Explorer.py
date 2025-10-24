# AI_Decision_Explorer.py - COMPLETE VERSION
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

print("🔬 ANALYZING AI'S DECISION PROCESS...")
print("Loading and training AI model...")

# Load data
digits = load_digits()

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(digits.data, digits.target)

print("✅ Model trained successfully!")
print("Calculating feature importance...")

# Calculate which features (pixels) are most important
result = permutation_importance(model, digits.data, digits.target, 
                               n_repeats=10, random_state=42)

# Reshape importance to image format
pixel_importance = result.importances_mean.reshape(8, 8)

print("Creating visualizations...")

# Create visualization
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# Original digit
ax1.imshow(digits.images[0], cmap='gray')
ax1.set_title("Human View\n(Digit Image)")
ax1.axis('off')

# AI's focus
ax2.imshow(pixel_importance, cmap='YlOrRd')
ax2.set_title("AI's Focus Areas\n(Bright = Important)")
ax2.axis('off')

# Overlay: Human + AI view
ax3.imshow(digits.images[0], cmap='gray')
im = ax3.imshow(pixel_importance, cmap='YlOrRd', alpha=0.6)
ax3.set_title("Combined View\n(AI Focus Overlay)")
ax3.axis('off')

plt.colorbar(im, ax=ax3, label='Importance')
plt.tight_layout()
plt.show()

print("💡 The AI focuses on the SHAPE of digits, not specific pixels!")
print("🎯 Bright areas show where the AI pays the most attention")

input("Press Enter to see more AI insights...")

# Additional analysis
print("\n📊 ADDITIONAL AI INSIGHTS:")

# Show which digits are easiest/hardest for AI
predictions = model.predict(digits.data)
accuracy_by_digit = []

for digit in range(10):
    mask = digits.target == digit
    accuracy = np.mean(predictions[mask] == digits.target[mask])
    accuracy_by_digit.append(accuracy)
    print(f"Digit {digit}: {accuracy:.1%} accuracy")

# Plot accuracy by digit
plt.figure(figsize=(10, 5))
bars = plt.bar(range(10), accuracy_by_digit, color='skyblue')
plt.xlabel('Digit')
plt.ylabel('Accuracy')
plt.title('AI Accuracy for Each Digit')
plt.xticks(range(10))
plt.grid(True, alpha=0.3)

# Color the bars based on accuracy
for bar, accuracy in zip(bars, accuracy_by_digit):
    if accuracy > 0.95:
        bar.set_color('green')
    elif accuracy < 0.85:
        bar.set_color('red')

plt.tight_layout()
plt.show()

print("🎉 Analysis complete! Your AI understands digit recognition!")
input("Press Enter to exit...")