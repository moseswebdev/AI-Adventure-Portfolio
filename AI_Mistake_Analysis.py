# Analyze where AI makes mistakes
from sklearn.metrics import confusion_matrix
import seaborn as sns

print("🔍 ANALYZING AI'S MISTAKES...")

# Get predictions
predictions = model.predict(digits.data)

# Find misclassifications
wrong_predictions = predictions != digits.target
wrong_indices = np.where(wrong_predictions)[0]

if len(wrong_indices) > 0:
    print(f"🤖 AI made {len(wrong_indices)} mistakes out of {len(digits.data)} samples")
    
    # Show some mistakes
    plt.figure(figsize=(15, 4))
    for i, idx in enumerate(wrong_indices[:5]):
        plt.subplot(1, 5, i + 1)
        plt.imshow(digits.images[idx], cmap='gray')
        plt.title(f"True: {digits.target[idx]}\nPred: {predictions[idx]}")
        plt.axis('off')
    
    plt.suptitle("Samples Where AI Made Mistakes", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(digits.target, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix: What Gets Confused with What')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
else:
    print("🎉 PERFECT! AI made ZERO mistakes on this dataset!")