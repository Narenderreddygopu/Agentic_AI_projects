from transformers import pipeline
classifier = pipeline('sentiment-analysis')
result = classifier("I love this course!")
print(result)