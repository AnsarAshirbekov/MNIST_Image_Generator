from flask import Flask, render_template, request, send_file
import torch
import matplotlib.pyplot as plt
import os
from generator3 import NoisePredictor, generate

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = NoisePredictor().to(device)
model.load_state_dict(
    torch.load("diffusion_mnist_model_v4.pth", map_location=device)
)
model.eval()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate_digit():
    digit = int(request.form["digit"])

    if digit < 0 or digit > 9:
        return "Введите число от 0 до 9"
    
    img = generate(digit)

    img = img.clamp(0, 1)
    img = img.cpu().detach()[0][0]

    os.makedirs("static/generated", exist_ok=True)

    path = f"static/generated/digit_{digit}.jpg"

    plt.imsave(path, img, cmap="gray")

    return render_template(
        "index.html",
        generated_image=path
    )


if __name__ == "__main__":
    app.run(debug=True)