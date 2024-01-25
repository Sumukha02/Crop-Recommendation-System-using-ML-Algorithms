from tkinter import *
from PIL import ImageTk, Image
import joblib
import os

# create the main window
root = Tk()
root.title("'Crop Yield Prediction - GUI'")

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative paths to the images
background_image_path = os.path.join(current_dir, "..", "util_imgs", "background_image.jpg")
input_bg_image_path = os.path.join(current_dir, "..", "util_imgs", "input_bg_image.jpeg")
output_bg_image_path = os.path.join(current_dir, "..", "util_imgs", "output_bg_image.jpeg")

# get the screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# resize the background image to fit the screen size
bg_image = Image.open(background_image_path)
bg_image = bg_image.resize((screen_width, screen_height), Image.ANTIALIAS)
bg_image = ImageTk.PhotoImage(bg_image)

# add the background image
bg_label = Label(root, image=bg_image)
bg_label.place(x=0, y=0, relwidth=1.0, relheight=1.0)

# increase the display size of the GUI
root.geometry("{}x{}".format(int(screen_width*0.9), int(screen_height*0.8)))

# add a label for the main title
title_label = Label(root, text="Crop Yield Prediction", font=("Courier", 30, "bold"), fg="#28a745")
title_label.place(relx=0.5, y=50, anchor="center")

# create a frame to hold the input fields and the predict button
input_frame = Frame(root, bg="white", bd=2, relief=SOLID)
input_frame.place(x=100, y=150, width=700, height=630)

# add a background image for the input frame
input_bg_image = Image.open(input_bg_image_path)
input_bg_image = input_bg_image.resize((700, 630), Image.ANTIALIAS)
input_bg_image = ImageTk.PhotoImage(input_bg_image)

input_bg_label = Label(input_frame, image=input_bg_image)
input_bg_label.place(x=0, y=0, relwidth=1.0, relheight=1.0)

# add input fields for the different factors
n_label = Label(input_frame, text="Nitrogen (N):", font=("Courier", 20, "bold"), fg="#28a745")
n_label.grid(row=0, column=0, padx=20, pady=20)

n_entry = Entry(input_frame, font=("Courier", 20))
n_entry.grid(row=0, column=1)

p_label = Label(input_frame, text="Phosphorus (P):", font=("Courier", 20, "bold"), fg="#28a745")
p_label.grid(row=1, column=0, padx=20, pady=20)

p_entry = Entry(input_frame, font=("Courier", 20))
p_entry.grid(row=1, column=1)

k_label = Label(input_frame, text="Potassium (K):", font=("Courier", 20, "bold"), fg="#28a745")
k_label.grid(row=2, column=0, padx=20, pady=20)

k_entry = Entry(input_frame, font=("Courier", 20))
k_entry.grid(row=2, column=1)

temp_label = Label(input_frame, text="Temperature (°C):", font=("Courier", 20, "bold"), fg="#28a745")
temp_label.grid(row=3, column=0, padx=20, pady=20)

temp_entry = Entry(input_frame, font=("Courier", 20))
temp_entry.grid(row=3, column=1)

humid_label = Label(input_frame, text="Humidity:", font=("Courier", 20, "bold"), fg="#28a745")
humid_label.grid(row=4, column=0, padx=20, pady=20)

humid_entry = Entry(input_frame, font=("Courier", 20))
humid_entry.grid(row=4, column=1)

ph_label = Label(input_frame, text="pH Value (0-14):", font=("Courier", 20, "bold"), fg="#28a745")
ph_label.grid(row=5, column=0, padx=20, pady=20)

ph_entry = Entry(input_frame, font=("Courier", 20))
ph_entry.grid(row=5, column=1)

rain_label = Label(input_frame, text="Rainfall:", font=("Courier", 20, "bold"), fg="#28a745")
rain_label.grid(row=6, column=0, padx=20, pady=20)

rain_entry = Entry(input_frame, font=("Courier", 20))
rain_entry.grid(row=6, column=1)

# create a frame to hold the Drop_Down label
DD_frame = Frame(root, bg="white", bd=2, relief=SOLID)
DD_frame.place(x=900, y=150, width=600, height=150)

# add a background image for the Drop_Down frame
DD_bg_image = Image.open(output_bg_image_path)
DD_bg_image = DD_bg_image.resize((600, 200), Image.ANTIALIAS)
DD_bg_image = ImageTk.PhotoImage(DD_bg_image)
DD_bg_label = Label(DD_frame, image=DD_bg_image)
DD_bg_label.place(x=0, y=0, relwidth=1.0, relheight=1.0)

# add a drop-down list to select the model
model_label = Label(DD_frame, text="Select Model:", font=("Courier", 20, "bold"), fg="#28a745")
model_label.grid(row=7, column=0, padx=20, pady=20)

selected_model = StringVar(DD_frame)
selected_model.set("Random Forest Classifier") # default value

model_options = ["Random Forest Classifier", "K-Nearest Neighbors", "Support Vector Machine", "Gradient Boosting Classifier"]
model_dropdown = OptionMenu(DD_frame, selected_model, *model_options)
model_dropdown.config(font=("Courier", 20), bg="white")
model_dropdown.grid(row=8, column=0)

# add a button to predict the crop yield
def predict_yield_gui():

    # # load the saved model
    # model = joblib.load("crop_yield_model.joblib")

    # load the selected model
    global model
    if selected_model.get() == "Random Forest Classifier":
        model = joblib.load(os.path.join(current_dir, "..", "models", "random_forest_model_yield.joblib"))
    elif selected_model.get() == "K-Nearest Neighbors":
        model = joblib.load(os.path.join(current_dir, "..", "models", "knn_model_yield.joblib"))
    elif selected_model.get() == "Support Vector Machine":
        model = joblib.load(os.path.join(current_dir, "..", "models", "svc_model_yield.joblib"))
    elif selected_model.get() == "Gradient Boosting Classifier":
        model = joblib.load(os.path.join(current_dir, "..", "models", "gradient_boost_model_yield.joblib"))

    # get the input values
    temp = float(temp_entry.get())
    rain = float(rain_entry.get())
    N = float(n_entry.get())
    P = float(p_entry.get())
    K = float(k_entry.get())
    humid = float(humid_entry.get())
    ph = float(ph_entry.get())

    # create a list to hold the input values
    input_data = [[N, P, K, temp, humid, ph, rain]]

    # make predictions on the input data
    predicted_yield = model.predict(input_data)

    # display the predicted crop yield
    yield_label.config(text="Predicted Crop Yield: {}".format(predicted_yield[0]))

def predict_yield_type_gui():

    # load the selected model
    global model
    if selected_model.get() == "Random Forest Classifier":
        model = joblib.load(os.path.join(current_dir, "..", "models", "random_forest_model_yield_type.joblib"))
    elif selected_model.get() == "K-Nearest Neighbors":
        model = joblib.load(os.path.join(current_dir, "..", "models", "knn_model_yield_type.joblib"))
    elif selected_model.get() == "Support Vector Machine":
        model = joblib.load(os.path.join(current_dir, "..", "models", "svc_model_yield_type.joblib"))
    elif selected_model.get() == "Gradient Boosting Classifier":
        model = joblib.load(os.path.join(current_dir, "..", "models", "gradient_boost_model_yield_type.joblib"))

    # get the input values
    temp = float(temp_entry.get())
    rain = float(rain_entry.get())
    N = float(n_entry.get())
    P = float(p_entry.get())
    K = float(k_entry.get())
    humid = float(humid_entry.get())
    ph = float(ph_entry.get())

    # create a list to hold the input values
    input_data = [[N, P, K, temp, humid, ph, rain]]

    # make predictions on the input data
    predicted_yield_type = model.predict(input_data)

    # display the predicted crop yield
    yield_type_label.config(text="Predicted Crop Yield_Type: {}".format(predicted_yield_type[0]))

def predict_gui():
    predict_yield_gui()
    predict_yield_type_gui()

predict_button = Button(input_frame, text="Predict", command=predict_gui, bg="#28a745", fg="white", font=("Courier", 20, "bold"))
predict_button.grid(row=7, column=0, columnspan=2, pady=20)

# create a frame to hold the output label
output_frame = Frame(root, bg="white", bd=2, relief=SOLID)
output_frame.place(x=900, y=550, width=600, height=170)

# add a background image for the output frame
output_bg_image = Image.open(output_bg_image_path)
output_bg_image = output_bg_image.resize((600, 200), Image.ANTIALIAS)
output_bg_image = ImageTk.PhotoImage(output_bg_image)
output_bg_label = Label(output_frame, image=output_bg_image)
output_bg_label.place(x=0, y=0, relwidth=1.0, relheight=1.0)

# add a label to display the predicted crop yield
yield_label = Label(output_frame, text="Predicted Crop Yield: ", font=("Courier", 20, "bold"), fg="#28a745")
yield_label.place(relx=0.5, rely=0.3, anchor="center")

# add a label to display the predicted crop yield_type
yield_type_label = Label(output_frame, text="Predicted Crop Yield_Type: ", font=("Courier", 20, "bold"), fg="#28a745")
yield_type_label.place(relx=0.5, rely=0.6, anchor="center")

# start the main loop
root.mainloop()