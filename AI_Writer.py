# story_generator.py
import random

print("📖 AI STORY GENERATOR")
print("=" * 40)

# AI learns to combine words creatively
subjects = ['The astronaut', 'The mysterious cat', 'The brave programmer', 'The lost robot']
verbs = ['discovered', 'created', 'solved', 'explored']
objects = ['a secret algorithm', 'a cosmic mystery', 'an AI companion', 'ancient code']

def generate_story():
    story = f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)}."
    return story

print("🤖 AI-GENERATED STORIES:")
for i in range(5):
    print(f"Story {i+1}: {generate_story()}")

print(f"\n🎯 Your AI can now create content!")
input("Press Enter for next level...")