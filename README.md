# LokDrishti

## Folder setup
Place the extracted drive-download contents inside a `data/` folder next to `app.py`.

Example:

lokdrishti_refined/
- app.py
- requirements.txt
- src/
- assets/
- data/
  - api_data_aadhar_enrolment/
  - api_data_aadhar_demographic/
  - api_data_aadhar_biometric/
  - rural-urban/
  - lgd urban.csv
  - Zones.csv

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- The application includes national, state, district, forecasting, urban-rural, capacity, clustering, and data-explorer views.
- The India geojson is cached under `assets/india_state.geojson` when available.
- If the geojson cannot be loaded at runtime, the state map view falls back to a chart-based view.
