# Fake Review Detector (Python Lab Project)

Checks if a product review looks fake (computer-generated) or genuine.

## Files

| File | What it is |
|---|---|
| `notebooks/fake_review_training.ipynb` | Colab notebook: loads data, makes charts, trains the model, saves `models/model.pkl` |
| `utils.py` | Helper functions used by the notebook and the app |
| `app.py` | Streamlit website |
| `requirements.txt` | Libraries for the website |
| `sample_product_reviews.csv` | 12 real reviews to try in the app |

## How to run

1. Google Drive: put `utils.py` in `MyDrive/fake-review-detector/` and the Kaggle file in `MyDrive/fake-review-detector/data/fake reviews dataset.csv`.
2. Open the notebook in Colab and click Runtime -> Run all.
3. Download the `models` folder from Drive and put it next to `app.py`.
4. On your laptop, in this folder:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
