#pip install numpy==1.26.0
#pip install opencv-python
#pip install cmake
#pip install dlib for python 3.11
#pip install face_recognition
#pip install openpyxl


import os
import cv2
import face_recognition
import pandas as pd
from datetime import datetime
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image as KivyImage
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.textinput import TextInput
from kivy.properties import StringProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.floatlayout import FloatLayout

# Function to load person details from a text file into a dictionary.
def load_person_details(file_path):
    details = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into individual attributes
            name, age, adhar,license, date_of_issue, date_of_expiry, status, image_path = line.strip().split(',')
            details[image_path] = {
                'name': name,
                'age': age,
                'adhar': adhar,
                'license': license,
                'date_of_issue': date_of_issue,
                'date_of_expiry': date_of_expiry,
                'status': status
            }
    return details

# Function to load known faces from images in a directory and encode them for face recognition.
def load_known_faces(directory_path):
    known_faces = {}
    for filename in os.listdir(directory_path):
        # Check if the file is an image
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            # Ensure encodings are found before adding
            if encodings:
                known_faces[filename] = encodings[0]  # Store the first encoding found
    return known_faces

# Define the logo screen of the app.
class LogoScreen(Screen):
    def __init__(self, **kwargs):
        super(LogoScreen, self).__init__(**kwargs)
        self.layout = FloatLayout()
        self.logo_image = KivyImage(source='logo.png',
                                     allow_stretch=False,
                                     keep_ratio=False,
                                     size_hint=(1, 1),
                                     pos_hint={'center_x': 0.5, 'center_y': 0.5})
        self.layout.add_widget(self.logo_image)
        # Schedule transition to the login screen after 9 seconds
        Clock.schedule_once(self.transition_to_login, 4)
        self.add_widget(self.layout)

    def transition_to_login(self, dt):
        self.manager.current = 'login'  # Change to the login screen

# Define the login screen of the app.
class LoginPage(Screen):
    def __init__(self, **kwargs):
        super(LoginPage, self).__init__(**kwargs)
        self.layout = FloatLayout()
        # Logo at the top of the login page
        self.logo_image = KivyImage(source='logo.png',
                                     allow_stretch=True,
                                     keep_ratio=False,
                                     size_hint=(None, None),
                                     size=(200, 200),
                                     pos_hint={'center_x': 0.5, 'top': 0.95})
        self.layout.add_widget(self.logo_image)
        
        # Title label for login
        self.title = Label(text='ADMIN LOGIN',
                           size_hint=(None, None),
                           size=(400, 60),
                           pos_hint={'center_x': 0.5, 'top': 0.7},
                           font_size='32sp',
                           bold=True,
                           color=(0, 0.5, 0.8, 1))
        self.layout.add_widget(self.title)

        # Username input field
        self.username = TextInput(hint_text='Username',
                                  size_hint=(None, None),
                                  size=(300, 50),
                                  pos_hint={'center_x': 0.5, 'center_y': 0.55},
                                  padding_y=(10, 10),
                                  background_normal='',
                                  background_color=(1, 1, 1, 1),
                                  foreground_color=(0, 0, 0, 1),
                                  border=(1, 1, 1, 1),
                                  font_size='18sp')
        self.layout.add_widget(self.username)

        # Password input field
        self.password = TextInput(hint_text='Password',
                                  password=True,
                                  size_hint=(None, None),
                                  size=(300, 50),
                                  pos_hint={'center_x': 0.5, 'center_y': 0.4},
                                  padding_y=(10, 10),
                                  background_normal='',
                                  background_color=(1, 1, 1, 1),
                                  foreground_color=(0, 0, 0, 1),
                                  border=(1, 1, 1, 1),
                                  font_size='18sp')
        self.layout.add_widget(self.password)

        # Login button
        self.login_button = Button(text='Login',
                                   size_hint=(None, None),
                                   size=(300, 50),
                                   pos_hint={'center_x': 0.5, 'center_y': 0.25},
                                   background_color=(0, 0.5, 1, 1),
                                   color=(1, 1, 1, 1),
                                   font_size='20sp',
                                   bold=True)
        self.login_button.bind(on_press=self.login)  # Bind the login function to the button
        self.layout.add_widget(self.login_button)

        # Error message label
        self.error_message = Label(size_hint=(None, None),
                                   size=(300, 30),
                                   pos_hint={'center_x': 0.5, 'center_y': 0.15},
                                   color=(1, 0, 0, 1),
                                   font_size='16sp',
                                   halign='center')
        self.layout.add_widget(self.error_message)
        self.add_widget(self.layout)

    def login(self, instance):
        username = self.username.text.strip()
        password = self.password.text.strip()

        # Validate username and password
        if not username or not password:
            self.error_message.text = 'Username and Password are required!'
        elif username == 'Shweta' and password == '123':  # Hardcoded credentials for simplicity
            self.manager.current = 'main'  # Transition to the main screen
        else:
            self.error_message.text = 'Invalid username or password!'

# Define the face detection screen of the app.
class FaceDetectionScreen(Screen):
    def __init__(self, **kwargs):
        super(FaceDetectionScreen, self).__init__(**kwargs)
        self.person_details = load_person_details('assets/person_details.txt')  # Load person details
        self.known_faces = load_known_faces('assets')  # Load known faces

        Window.clearcolor = (0.9, 0.9, 0.9, 1)  # Set window background color

        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        # Title for face detection app
        title_label = Label(text="LICENSE DETECTOR APP",
                            size_hint=(1, 0.1),
                            color=(0, 0.5, 0.8, 1),
                            bold=True, font_size='30sp')
        self.layout.add_widget(title_label)

        # Box for camera feed
        self.camera_box = BoxLayout(orientation='vertical', size_hint=(1, 1))
        self.image = KivyImage(size_hint=(1, 0.8))
        self.camera_box.add_widget(self.image)
        self.layout.add_widget(self.camera_box)

        # Label to show detection status
        self.result_label = Label(text="PRESS THE BUTTON TO OPEN CAMERA.",
                                  size_hint=(1, 0.1),
                                  color=(0, 0, 0, 1),
                                  bold=True, font_size='24sp')
        self.layout.add_widget(self.result_label)

        # Button to open the camera
        self.open_camera_button = Button(text="OPEN CAMERA",
                                         size_hint=(0.5, 0.1),
                                         pos_hint={'center_x': 0.5, 'center_y': 0.5},
                                         background_color=(0, 0.8, 0.8, 1),
                                         color=(1, 1, 1, 1),
                                         bold=True, font_size='24sp')
        self.open_camera_button.bind(on_press=self.start_camera)  # Bind camera start function
        self.layout.add_widget(self.open_camera_button)

        # Button to show detected person details
        self.show_details_button = Button(text="SHOW DETAILS",
                                           size_hint=(0.5, 0.1),
                                           pos_hint={'center_x': 0.5, 'center_y': 0.5},
                                           background_color=(0, 0.8, 0.8, 1),
                                           color=(1, 1, 1, 1),
                                           bold=True, font_size='24sp')
        self.show_details_button.bind(on_press=self.show_details)  # Bind show details function
        self.show_details_button.opacity = 0  # Initially hidden
        self.layout.add_widget(self.show_details_button)

        self.add_widget(self.layout)

        # Initialize variables for detected people
        self.detected_people_data = []
        self.detected_people_set = set()
        self.current_detected_info = None
        self.details_found = True

    def start_camera(self, instance):
        self.capture = cv2.VideoCapture(0)  # Start video capture from the camera
        Clock.schedule_interval(self.update, 1.0 / 15.0)  # Update frame every 1/15 seconds
        self.open_camera_button.opacity = 0  # Hide the open camera button
        self.show_details_button.opacity = 0  # Hide the show details button
        self.result_label.text = "DETECTING..."  # Update label to show detecting status

    def stop_camera(self, instance):
        if hasattr(self, 'capture') and self.capture.isOpened():
            self.capture.release()  # Release the camera
        self.image.texture = None  # Clear the image texture
        self.open_camera_button.opacity = 1  # Show the open camera button
        self.show_details_button.opacity = 0  # Hide the show details button
        self.result_label.text = "PRESS THE BUTTON TO OPEN CAMERA."  # Reset status message
        self.save_to_excel()  # Save detected people data to Excel

    def update(self, dt):
        ret, frame = self.capture.read()  # Read a frame from the camera
        if ret:
            frame = cv2.resize(frame, (800, 600))  # Resize the frame
            face_encodings = face_recognition.face_encodings(frame)  # Find face encodings in the frame

            if face_encodings:
                self.result_label.text = "FACE DETECTED"  # Update status message
                for face_encoding in face_encodings:
                    face_encoding_hash = str(face_encoding.tolist()).strip()  # Convert encoding to string for comparison

                    # Check if the face has already been detected
                    if face_encoding_hash not in self.detected_people_set:
                        self.detected_people_set.add(face_encoding_hash)  # Add to detected set
                        match_found = False
                        for known_image_path, known_encoding in self.known_faces.items():
                            results = face_recognition.compare_faces([known_encoding], face_encoding)  # Compare with known faces
                            if results[0]:  # If a match is found
                                match_found = True
                                person_info = self.person_details.get(known_image_path, {
                                    'name': 'Unknown',
                                    'age': 'Unknown',
                                    'adhar': 'Unknown',
                                    'license': 'Unknown',
                                    'date_of_issue': 'Unknown',
                                    'date_of_expiry': 'Unknown',
                                    'status': 'Unknown'
                                })
                                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get current timestamp

                                # Create a string with the detected person's information
                                self.current_detected_info = (
                                    f"NAME: {person_info['name']}\n"
                                    f"AGE: {person_info['age']}\n"
                                    f"AADHAR: {person_info['adhar']}\n"
                                    f"LICENSE: {person_info['license']}\n"
                                    f"DATE OF ISSUE: {person_info['date_of_issue']}\n"
                                    f"DATE OF EXPIRY: {person_info['date_of_expiry']}\n"
                                    f"STATUS: {person_info['status']}\n"
                                    f"TIME: {timestamp}"
                                )
                                self.details_found = True
                                self.show_details_button.opacity = 1  # Show details button

                                # Save the details immediately
                                self.save_details_to_excel(self.current_detected_info)  # Save details here
                                break

                        if not match_found:
                            self.result_label.text = "DETAILS NOT FOUND"  # If no match found
                            self.details_found = False
            else:
                self.result_label.text = "DETAILS NOT FOUND"  # No faces detected
                self.details_found = False

            # Convert the frame to a texture for Kivy
            buf1 = cv2.flip(frame, -1)  # Flip the frame
            buf = buf1.tobytes()  # Convert to bytes
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')  # Load bytes into texture
            self.image.texture = image_texture  # Update the image widget

    def save_details_to_excel(self, details):
        # Convert details string to a dictionary
        details_list = details.split("\n")
        details_dict = {item.split(": ")[0]: item.split(": ")[1] for item in details_list if ": " in item}

        # Create a DataFrame from the details dictionary
        df = pd.DataFrame([details_dict])

        # Write to the Excel file immediately
        file_path = 'detected_people.xlsx'
        if not os.path.exists(file_path):
            df.to_excel(file_path, index=False)  # Create new file if it doesn't exist
        else:
            with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                df.to_excel(writer, index=False, header=False)  # Append without header

    def show_details(self, instance):
        # Get the details screen and display the detected person's details
        details_screen = self.manager.get_screen('details')
        if self.details_found:
            details_screen.display_details(self.current_detected_info)
        else:
            details_screen.display_details("DETAILS NOT FOUND")
        self.manager.current = 'details'  # Transition to the details screen

    def save_to_excel(self):
        # Save detected people data to Excel if available
        if self.detected_people_data:
            df = pd.DataFrame(self.detected_people_data)
            df.to_excel('detected_people.xlsx', index=False, engine='openpyxl')

    def on_stop(self):
        # Release the camera and save data when the app is closed
        if hasattr(self, 'capture') and self.capture.isOpened():
            self.capture.release()
        self.save_to_excel()

# Define the details screen of the app.
class DetailsScreen(Screen):
    def __init__(self, **kwargs):
        super(DetailsScreen, self).__init__(**kwargs)
        self.layout = FloatLayout()

        # Title for details screen
        self.title_label = Label(
            text='DETAILS',
            size_hint=(1, 0.1),
            pos_hint={'center_x': 0.5, 'top': 0.95},
            font_size='50sp',
            bold=True,
            color=(0, 0.5, 0.8, 1)
        )
        self.layout.add_widget(self.title_label)

        # Label to display the details
        self.details_label = Label(
            size_hint=(0.8, 0.6),
            pos_hint={'x': 0.28, 'center_y': 0.5},  # Adjust the 'x' value for margin
            color=(0, 0, 0, 1),
            font_size='30sp',
            halign='left',
            valign='middle',
            text_size=(self.width * 0.8 - 20, None)  # Adjust the text size if needed
        )

        self.details_label.bind(size=self.details_label.setter('text_size'))  # Bind size to text size
        self.layout.add_widget(self.details_label)

        # Button to close the camera
        self.close_camera_button = Button(
            text="CLOSE CAMERA",
            size_hint=(0.3, 0.1),
            pos_hint={'center_x': 0.5, 'bottom': 0.1},
            background_color=(1, 0, 0, 1),
            color=(1, 1, 1, 1),
            bold=True, font_size='24sp'
        )
        self.close_camera_button.bind(on_press=self.close_camera)  # Bind close function
        self.layout.add_widget(self.close_camera_button)

        self.add_widget(self.layout)

    def display_details(self, details):
        self.details_label.text = details  # Set the details text

    def close_camera(self, instance):
        # Stop the camera and go back to the main screen
        face_detection_screen = self.manager.get_screen('main')
        face_detection_screen.stop_camera(instance)
        self.manager.current = 'main'  # Transition to the main screen

# Define the main app class.
class FaceDetectionApp(App):
    def build(self):
        sm = ScreenManager()  # Create a ScreenManager to manage different screens
        sm.add_widget(LogoScreen(name='logo'))  # Add logo screen
        sm.add_widget(LoginPage(name='login'))  # Add login page
        sm.add_widget(FaceDetectionScreen(name='main'))  # Add face detection screen
        sm.add_widget(DetailsScreen(name='details'))  # Add details screen
        sm.current = 'logo'  # Set initial screen to logo
        return sm

# Run the app.
if __name__ == '__main__':
    FaceDetectionApp().run()
