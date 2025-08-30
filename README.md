# Visual Product Matcher  

An AI-powered web application that helps find visually similar products based on an uploaded image or image URL.  
This project demonstrates how computer vision can be applied in e-commerce to improve product discovery and recommendations.  

---

## 🚀 Features  
- Upload an image or paste an image URL to search.  
- Finds the top visually similar products from the dataset.  
- Precomputed product database for fast searching.  
- Simple and mobile-friendly UI.  
- Built with Flask and deep learning models.  

---

## 🛠️ Tech Stack  
- **Backend:** Python, Flask  
- **ML/DL:** PyTorch, OpenCLIP (image embeddings)  
- **Database:** SQLite (products.db)  
- **Frontend:** HTML, CSS, JavaScript  

---

## ⚡ How to Run Locally  

1. Clone this repository:
   ```bash
   git clone https://github.com/vithikagupta411/visual-product-matcher.git
   cd visual-product-matcher
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Precompute the product embeddings (this creates products.db):

bash
Copy code
python prepare_data.py
Start the Flask server:

bash
Copy code
python app.py
Open the app in your browser:

cpp
Copy code
http://127.0.0.1:5000
🌍 Deployment
This project is deployed on Render.
🔗 Live Link: Click Here

👩‍💻 Author
Vithika Gupta

📧 Email: vithikaguptaa411@gmail.com

💼 LinkedIn: linkedin.com/in/vithika-gupta-328a61261

💻 GitHub: github.com/vithikagupta411

📌 Notes

The dataset (products.csv) contains sample product entries with categories and image URLs.

Future improvements can include larger datasets and better ranking methods.