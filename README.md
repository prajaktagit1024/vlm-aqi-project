# AQI Prediction with Vision-Language Model (CLIP)

Predicts Air Quality Index from sky/pollution images using CLIP + a regression head.

## Project Structure

```
aqi-vlm/
├── data/
│   ├── generate_sample_data.py   # generates synthetic dataset
│   └── images/                   # generated sky images
├── src/
│   ├── dataset.py                # data loading & CLIP preprocessing
│   ├── model.py                  # CLIP + regression head
│   ├── train.py                  # training pipeline
│   ├── evaluate.py               # metrics & plots
│   └── inference.py              # predict from new image
├── checkpoints/                  # saved model weights
├── plots/                        # evaluation plots
├── app.py                        # Streamlit web app
└── requirements.txt
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic dataset
python data/generate_sample_data.py

# 3. Train the model
python src/train.py

# 4. Evaluate on test set
python src/evaluate.py

# 5. Predict from a new image
python src/inferen
ce.py --image path/to/sky.jpg --text "Hazy morning, 35C"

# 6. Run the web app
streamlit run app.py
```

## Using Real Data

### Option A: AQI API + Image Scraping
1. Get free API key from [OpenAQ](https://openaq.org/) or [AirVisual](https://www.iqair.com/air-pollution-data-api)
2. Fetch historical AQI readings with timestamps and locations
3. Match timestamps to sky images from weather cams or Flickr/Unsplash by location/date
4. Use `requests` + `BeautifulSoup` to scrape images

### Option B: Existing Datasets
- [RESIDE dataset](https://sites.google.com/view/reside-dehaze-datasets) - hazy/clear image pairs
- [PM2.5-GAN dataset](https://github.com/zhenglab/PM2.5-GAN) - pollution images with PM2.5 labels
- [Beijing PM2.5 dataset](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data) - tabular + can be paired with images

## Model Architecture

```
Image → CLIP ViT-B/32 → [512-dim embedding]
                                              → Concat → MLP → AQI
Text  → CLIP Text Enc → [512-dim embedding]
```

- CLIP backbone is frozen (only regression head trains by default)
- Top 2 CLIP vision layers are unfrozen for fine-tuning
- Huber loss for robustness to AQI outliers
- Output normalized to [0,1], multiplied by 500 for real AQI

## Deployment Ideas

- **Docker + FastAPI**: wrap `inference.py` in a REST endpoint
- **AWS Lambda + S3**: serverless inference, store images in S3
- **ONNX export**: convert model for faster CPU inference
- **Mobile**: export to TFLite or CoreML for on-device prediction
- **IoT integration**: pair with a camera module + Raspberry Pi for real-time monitoring
