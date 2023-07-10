import os
import openai
import gradio as gr
from pytesseract import image_to_string
from PIL import Image
import cv2
import numpy as np
from pdf2image import convert_from_path
import tempfile
import asyncio

# Setup the asyncio loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# OpenAI configuration
openai.organization = "org-Hhs0eGEn8PnZG75ixGcPUcTj"
openai.api_key = "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"


def preprocess_image(image_path):
    if not os.path.isfile(image_path):
        print(f"File does not exist: {image_path}")
        return

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Failed to load image file: {image_path}")
        return

    # New steps: noise removal and contrast enhancement
    blur = cv2.medianBlur(image, 3)  # Median blur for noise removal
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(blur)

    # Original steps with slight modification (using the enhanced image)
    blur = cv2.GaussianBlur(image, (5,5), 0)
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((1, 1), np.uint8)
    img_dilation = cv2.dilate(threshold, kernel, iterations=1)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=1)

    coords = np.column_stack(np.where(img_erosion > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    cv2.imwrite(image_path, deskewed)


def process_file(file_path):
    if file_path.lower().endswith('.pdf'):
        images = convert_from_path(file_path)
        result = ""
        for i in range(len(images)):
            # Convert PIL Image to cv2 Image
            open_cv_image = np.array(images[i]) 
            # Convert RGB to BGR 
            open_cv_image = open_cv_image[:, :, ::-1].copy() 
            # Save the processed image to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            cv2.imwrite(temp_file.name, open_cv_image)
            # Preprocess image
            preprocess_image(temp_file.name)
            result += image_to_string(temp_file.name, lang='eng', config='--psm 3')
            temp_file.close()  # Important to close the file so we can read from it later in Windows
            os.unlink(temp_file.name)  # Delete the temporary file
    else:
        # Preprocess image
        preprocess_image(file_path)
        result = image_to_string(file_path, lang='eng', config='--psm 3')

    return f"The medical document says: {result}"

def openai_chat(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response


def upload_and_chat(file, input, history):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    if file is not None:
        # Process the uploaded file
        processed_file_text = process_file(file.name)

        # Create a new OpenAI ChatCompletion to correct the OCR errors
        prompt = f"The following is a medical document that has been processed with OCR. Some errors have occurred during OCR processing. Please correct any errors and provide a clean version of the document: {processed_file_text}"
        corrected_file_text = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant that corrects OCR errors."}, {"role": "user", "content": prompt}]
        )['choices'][0]['message']['content']

        initial_message = [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": corrected_file_text}
        ]

        if not history:
            history = initial_message
    else:
        # Process the chat input
        history.append({"role": "user", "content": input})

    response = openai_chat(history)

    # Get the assistant's reply
    output = response['choices'][0]['message']['content']

    # Append the assistant's reply to the history
    history.append({"role": "assistant", "content": output})

    # Convert the history to the format expected by gradio.Chatbot
    gradio_history = [(message["role"], message["content"]) for message in history]

    return gradio_history, history




chat_interface = gr.Interface(fn=upload_and_chat, 
                              inputs=[gr.File(label="Upload and Start Chat"), gr.Textbox(placeholder="Type your message here"), gr.State(initial=[])],
                              outputs=[gr.Chatbot(), gr.State()],
                              title="Chat with VA Medical Assistant",
                              description="Upload a PDF or image file with medical text and ask questions about the uploaded medical records or any health concerns.")

chat_interface.allow_outgoing_network = True
chat_interface.launch(debug=True)



