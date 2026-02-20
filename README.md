# 🏆 Datathon 2026

Welcome to the **Datathon 2026**! This repository contains all the data, guidelines, and starter code you need to compete. Read everything carefully before getting started.

> [!IMPORTANT]
> Teams should consist of a **maximum of 4 participants**. Solo participants are permitted.

---

## 📊 Tracks

This datathon consists of **three independent tracks**. You may compete in one or more.

| Track | Problem Type | Evaluation |
|-------|-------------|------------|
| [Track 1 – Image Classification](#-track-1--image-classification) | Supervised Learning | Macro F1 |
| [Track 2 – Text Classification](#-track-2--text-classification) | Supervised Learning | Micro F1 |
| [Track 3 – Urban Data Analysis](#-track-3--urban-data-analysis) | Open-ended Analysis | Judged Panel |

---

## 🖼️ Track 1 – Image Classification

Fine-grained image classification across **102 categories**. Your model must predict the correct class label for each image in the withheld test set. Organizers will run your `predict.py` on the test set after submission.

**Data is located in:** `participants/track1/`

```
participants/track1/
├── train/images/       ← training images (flat folder)
├── val/images/         ← validation images (flat folder, clean labels)
├── train.csv           ← image_id, filename, label
├── val.csv             ← image_id, filename, label
├── label_list.txt      ← all valid class labels
└── predict.py          ← implement this and submit it
```

> [!NOTE]
> Training labels contain noise. Use your judgment when designing your training pipeline.

**Categories:**
- **Category 1** – Pretrained vision model weights allowed
- **Category 2** – All weights trained from scratch only

📄 [Full Track 1 Guidelines](participants/track1/Participant%20Guidelines.md)

---

## 📰 Track 2 – Text Classification

Multi-label text classification on a collection of newswire articles. Each article may belong to one or more topic categories. Organizers will run your `predict.py` on the withheld test set after submission.

**Data is located in:** `participants/track2/`

```
participants/track2/
├── train.csv           ← article_id, title, text, topics
├── val.csv             ← article_id, title, text, topics
├── label_list.txt      ← all valid topic labels
└── predict.py          ← implement this and submit it
```

**Categories:**
- **Category 1** – Pretrained word embeddings allowed (Word2Vec, GloVe, FastText, etc.)
- **Category 2** – All representations learned from scratch only

📄 [Full Track 2 Guidelines](participants/track2/Participant%20Guidelines.md)

---

## 🏙️ Track 3 – Urban Data Analysis

Exploratory analysis, visualization, and predictive modeling using real-world municipal data from the City of Syracuse. Unlike Tracks 1 and 2, this track is open-ended and judged by a panel.

**Available Datasets:**
- [Assessment Data](https://data.syr.gov/search?layout=grid&tags=assessment)
- [Tax Parcels & Parcel Geometry](https://data.syr.gov/search?tags=tax%2520parcels%2Cparcels%2Cquarterly%2520parcel%2520download%2Cparcel)
- [Budget Datasets](https://data.syr.gov/search?q=budget)
- [Parking Violations](https://data.syr.gov/datasets/ed3bd67233154117ad894ce4f2430f5c_0/explore)
- [Unfit Properties](https://data.syr.gov/datasets/71525e1b176e4fcba2edee8c6a590f84_0/explore)
- [Code Violations](https://data.syr.gov/datasets/107745f070b049feb38273a7ab200487_0/explore)
- [Vacant Properties](https://data.syr.gov/datasets/c23882a4f7904747ab7a4cab637d912d_0/explore)
- [Crime Datasets](https://data.syr.gov/search?layout=grid&tags=crime)

**Judging Awards (25 pts each):** Best Insight · Best Trend · Best Visualization · Best Prediction

---

## ⚙️ Getting Started

### Option 1: Fork & Clone
1. Click the **Fork** button in the top-right corner
2. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
```
3. Share the repo with your team

### Option 2: Download Directly
Navigate to any file, click it, and use the download icon in the top-right corner.

---

## 📏 Rules

> [!IMPORTANT]
> Violations will result in **disqualification**.

- ❌ No large language models in any form — open source, API-based, or instruction-tuned
- ❌ No external datasets — use only the data provided for each track
- ✅ All work must be your own — cite any code not written by you
- ✅ Teams of up to 4 people
- ✅ All submissions must be fully reproducible

---

## 📑 Submission Instructions

### Tracks 1 & 2
Submit a single zip file per track:

```
teamname_track[1or2]_cat[1or2].zip
├── predict.py        ← paths must be set to test.csv before submitting
├── train.py
├── model/
├── requirements.txt
└── report.pdf        ← max 2 pages
```

Organizers will run:
```bash
pip install -r requirements.txt
python predict.py
```

### Track 3
```
teamname_track3.zip
├── report.pdf
├── visualizations/
└── code/
```

**Submit to:** *[Submission link — provided at event start]*

---

## 🗓️ Schedule

### Day 1
| Time | Event |
|------|-------|
| 10:00 AM | Check In 📝 |
| 11:30 AM | Opening Ceremony 🎤 |
| 12:00 PM | Hacking Begins ⏰ |

### Day 2
| Time | Event |
|------|-------|
| 12:00 PM | Hacking Ends ⏰ |
| 1:00 PM | Judging Begins 📝 |
| 4:00 PM | Closing Ceremony & Winners Announced 🏅 |

---

## 🔥 Prizes

*Prize details will be announced at the opening ceremony.*

---

> [!TIP]
> 💡 **Need help?** Reach out to an organizer at any time. Good luck! 🚀