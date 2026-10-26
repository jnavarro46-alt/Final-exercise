# Campus Facility Reservation Monitoring (Streamlit)
Deploys the Midterm Decision Tree (cancellation/no-show prediction).
Run: `pip install -r requirements.txt && streamlit run app.py`
Retrain (once): `python make_sample_data.py && python train_model.py`
Put your real CSV in `data/facility_reservations.csv`, and your notebook at `midterm/Midterm_Notebook.ipynb`.
Deploy: push to GitHub -> https://share.streamlit.io -> app.py.
Data is synthetic. Predictions are decision support, not automatic denial of service.
