
!pip install google-cloud-vision

"""**Main Script**"""

import io
import os
import pandas as pd

# Set the Google Cloud API key as an environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/content/key.json"

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision_v1 import types

# Instantiates a client
client = vision.ImageAnnotatorClient()

file_name = "/content/cat.png"

# Loads the image into memory
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)

# Perform label detection on the image file
response = client.label_detection(image=image)
labels = response.label_annotations

"""**Export the array list to a pandas dataframe**"""

df = pd.DataFrame(columns=['description', 'score'])

for label in labels:
    df = df.append(
        dict(
            description=label.description,
            score=label.score,
        ), ignore_index=True)
    
print(df)

"""**Save the df as csv to be used by the front end**"""

df.to_csv('outputDoc.csv')
