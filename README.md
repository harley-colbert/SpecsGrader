Certainly! Here is a **full, detailed README.md** tailored for your app, based on the logic and code you’ve shared, your workflow, and standard best practices for such an application.

---

# Risk & Review Department Classifier

This application provides a **first-pass AI-powered classification** of engineering or project specification statements into appropriate “Risk Level” and “Review Department” categories. The app is designed for rapid triage of large specification datasets, using **ensemble machine learning and semantic similarity** approaches, and is intended to speed up human review and department assignment.

---

## Table of Contents

* [Purpose](#purpose)
* [Features](#features)
* [How It Works](#how-it-works)
* [How To Use](#how-to-use)
* [Installation & Setup](#installation--setup)
* [Typical Workflow](#typical-workflow)
* [Technical Details](#technical-details)
* [Human Review Step](#human-review-step)
* [Troubleshooting](#troubleshooting)
* [Customization](#customization)
* [Acknowledgements](#acknowledgements)

---

## Purpose

* **Automate the initial assignment** of "Risk Level" and "Review Department" for hundreds or thousands of specification statements.
* Reduce manual effort by surfacing likely classifications and highlighting lines requiring further human attention.
* **All final results must be reviewed by a human**—this system is for *first pass, triage, and batch sorting* only.

---

## Features

* **Batch import** of CSV or Excel files with specification lines.
* **Multipass ensemble classification** using three approaches:

  * **Rule-based heuristics:** Keyword, phrase, and regex matching for high-confidence or manual review triggers.
  * **Classic ML:** Traditional scikit-learn text classification with confidence thresholds.
  * **Semantic Similarity:** Embedding-based nearest-neighbor match to prior labeled specs (cosine similarity).
* **Confidence-based adjudication:** The system picks the highest-confidence available label from rule, ML, or similarity.
* **Human-friendly output:** Detailed CSV with each statement, predicted labels, label source, and (optionally) nearest training text.
* **Model management:** Easily load, save, or switch between multiple model sets.
* **Verbose logging/debug mode** to help trace decisions.
* **Cross-platform:** Runs on Windows, Linux, and Mac with Python 3.8+.
* **Modern GUI**: (Tkinter/ttkbootstrap or PySide6, depending on your version.)

---

## How It Works

For each line of text (“Risk Description”):

1. **Rule Engine:**

   * Checks for high-risk or department keywords/phrases, too-short lines, or other heuristic triggers.
   * If a rule triggers, its forced label is used (e.g., “manual review” for 1-2 word lines).
2. **Classic ML:**

   * Runs trained scikit-learn models to predict risk level and department.
   * If prediction confidence is high enough, ML’s label is used.
3. **Semantic Similarity:**

   * Uses vector embeddings to find the most similar prior training statement and adopts its labels.
   * If neither rules nor classic ML are confident, this is used.
4. **Output Row:**

   * For each row, records: original statement, assigned risk, assigned department, source of label (“Rule”, “Classic”, “Similarity”), similarity score, and matching statement (if applicable).

---

## How To Use

### 1. **Train Models (Initial or Re-Train)**

* **Load a labeled CSV** of historical specs, each with “Risk Description”, “Risk Level”, and “Review Department” columns.
* Click **“Train Models”**—the system trains and saves an ensemble of classic ML and semantic embedding models.

### 2. **Classify New Data**

* **Select your latest spec document** (CSV or Excel, with risk description text).
* Click **“Classify (Multipass)”**—each row will be processed and auto-labeled.
* Results will display in the app and can be **saved to a new CSV**.

### 3. **(Optional) Review Most Similar Training Spec**

* Use the “Show Most Similar Training Spec” checkbox to display the most similar prior labeled text and similarity score for each result.

### 4. **Save/Load Model Sets**

* You can save and switch between model sets—useful if you have multiple projects/domains.

### 5. **Review Results**

* **ALWAYS perform a human review** of the output CSV.
* The tool is a *first pass* triage—final quality is ensured by your expertise.

---

## Installation & Setup

### **Dependencies**

* Python 3.8 or newer
* `pandas`, `joblib`, `scikit-learn`, `PySide6` **OR** `tkinter` and `ttkbootstrap`, `numpy`
* Project-specific modules: `embeddings.py`, `similarity_engine.py`, `classic_ml.py`, `rules_engine.py`, etc.

### **Setup Steps**

1. **Clone or unzip the app.**
2. *(Optional, recommended)*: Create and activate a new virtual environment.
3. **Install dependencies**
   `pip install pandas joblib scikit-learn numpy pyside6`
4. **Run the app:**

   * For GUI: `python splash.py` or `python app.py`
   * For CLI or debug: `python logic.py` with sample/test code enabled.

---

## Typical Workflow

1. **Obtain and clean your training data.**
2. **Train or update models** as your data grows.
3. **Batch classify new spec files.**
4. **Export and review the output**—use the “Label Source” and “Similarity Score” columns to identify low-confidence rows needing attention.
5. **(Critical) Human review:**

   * All auto-labeled data must be checked before use.
   * Use the “manual review/unclassified” and “similarity <0.5” rows as high-priority flags.

---

## Technical Details

* **Rule engine:** Assigns high-confidence or “manual review/unclassified” based on known words/phrases and heuristics.

* **Classic ML:** Uses scikit-learn classifiers and label encoders; only “high confidence” predictions are used.

* **Semantic Similarity:** Uses a vectorizer/embedding model to compare to historical training data; picks closest match (cosine similarity).

* **Row output includes:**

  * Risk Description
  * Final Risk Level
  * Final Review Department
  * Label Source (Rule, Classic, Similarity with score)
  * Similarity Match (if applicable)
  * Similarity Score (if applicable)

* **All rules and confidence thresholds** are easily modifiable in `rules_engine.py` and `logic.py`.

---

## Human Review Step

**This is a triage tool!**

* Never deploy results directly to production or use as official classifications without a thorough human review.
* The system is designed to reduce labor, but *final risk and department assignment is always a human decision*.

---

## Troubleshooting

* **Errors on launch:**

  * Check Python version and dependencies.
* **Blank/incorrect results:**

  * Make sure your input file has a “Risk Description” column.
  * Confirm you have trained and loaded models.
* **Bad confidence:**

  * Review training data quality; re-train if necessary.
  * Tweak rules in `rules_engine.py` to catch new keywords/phrases.

---

## Customization

* **To adjust risk/department keywords**: edit `rules_engine.py`.
* **To change classic ML models or thresholds**: see `classic_ml.py` and `logic.py`.
* **To tweak similarity cutoff/embedding model**: edit `similarity_engine.py`.

---

## Acknowledgements

* Based on open source libraries: pandas, scikit-learn, numpy, PySide6/Tkinter.
* System concept and interface by \[Your Name or Team].

---

## Questions?

* Contact \[your email] or open an issue in your project repo.

---

#HOW TO

Absolutely! Here is a **detailed "How To" guide** for each function and user interaction in your Risk & Department Classifier app. This guide covers both **end-user steps** and the reasoning/context for each function.

---

# How To Use: Risk & Review Department Classifier

This section provides **step-by-step, detailed instructions** for each user-facing function in the application. For each feature, you’ll find what it does, why you would use it, prerequisites, exact steps, and what to expect at each stage.

---

## 1. **Training New Models**

### What It Does

* Trains new machine learning and semantic similarity models on your labeled data.
* Required for up-to-date and accurate classification.

### Why Use It?

* You should **train (or retrain) models** whenever you have new, more complete, or better-labeled data.
* Ensures the system’s predictions match your latest project needs.

### Prerequisites

* A CSV file with at least three columns:

  * `Risk Description` (text to classify)
  * `Risk Level` (known/desired risk for each row)
  * `Review Department` (department for each row)
* The file should be **clean** (no missing data in required columns).

### Steps

1. Click **Select Training CSV**.

   * A file dialog opens. Navigate to your training file and select it.
2. Confirm your file is listed next to the button.
3. Click **Train Models**.

   * The app will process the file, train two sets of models (classic ML and semantic embedding), and display progress.
   * Training may take a few seconds to minutes depending on data size.
4. When training completes:

   * You’ll see a report in the output area.
   * The model set is loaded into the app.
5. (Recommended) **Save the model set** under a memorable name (e.g. "2024-Q2-Robotics").

   * Click **Save Model Set** and enter a name.
   * This lets you reload these models later, or keep different models for different projects.

### What To Expect

* After training, the app is ready to classify new data using the new models.
* The last trained model set is automatically loaded.

---

## 2. **Classifying New Data**

### What It Does

* Runs all new, unlabeled “Risk Description” text through your trained models.
* Assigns a risk level and review department to each row.

### Why Use It?

* Use this to **batch-assign departments and risk** for new specification documents, saving hours of manual triage.

### Prerequisites

* A CSV or Excel file with at least one column containing the text to classify (ideally named `Risk Description`).

### Steps

1. Click **Select File to Classify**.

   * A dialog opens. Select the file you want to process.
2. Ensure your model set is loaded (check the model dropdown/top label).
3. Optionally check **Show Most Similar Training Spec for Each** for richer output.
4. Click **Classify (Multipass)**.

   * The app will analyze each row and assign labels.
   * It decides for each line which method (rule, classic ML, similarity) is most reliable.
5. Review the results in the output area.

### What To Expect

* The output area will show a summary table.
* Each row includes the original description, assigned risk, department, label source, and (optionally) the closest matching spec and similarity score.

---

## 3. **Saving Results to CSV**

### What It Does

* Saves the classification output to a new CSV file.

### Why Use It?

* Allows you to export and review the assigned labels in Excel or other tools.
* Preserves all auto-assigned labels, scores, and match info.

### Steps

1. After classifying, click **Save Results to CSV**.
2. Choose a location and filename.
3. Confirm the export.
4. (Recommended) Open the exported CSV in Excel and filter/sort by:

   * **Label Source** (e.g. flag all “manual review/unclassified” for extra checking)
   * **Similarity Score** (flag low scores as uncertain)
   * **Risk/Department** for department-specific review

### What To Expect

* A CSV with all original descriptions, predictions, and label metadata.

---

## 4. **Saving/Loading Model Sets**

### What It Does

* Saves or reloads model sets (the full trained classifier and embeddings) under a name of your choice.

### Why Use It?

* Lets you maintain multiple project- or customer-specific classifiers.
* Easily switch between different models for different document sets.

### Steps (Saving)

1. After training, click **Save Model Set**.
2. Enter a descriptive name (e.g. "2024-BodyShop", "Q3-WeldCell").
3. The model set will appear in the dropdown and be stored for future sessions.

### Steps (Loading)

1. Use the **Model Set Dropdown** at the top.
2. Select any available set by name.

   * The models will load and be ready for classification.

### What To Expect

* Seamless switching between pre-trained models.
* No need to retrain for every classification session.

---

## 5. **Refreshing Model List**

### What It Does

* Updates the model set dropdown, in case new model sets have been added/saved by another process or user.

### Steps

1. Click **Refresh Model Sets**.
2. The dropdown will update with all available saved model sets.

---

## 6. **Configuring Output Options**

### “Show Most Similar Training Spec for Each”

* **What:** If enabled, every result will include the closest training example (statement), its labels, and the cosine similarity score.
* **Why:** Great for auditing the system and reviewing edge cases.
* **How:**

  1. Check or uncheck the option before running classification.
  2. Results will include (or omit) the similarity columns.

---

## 7. **Reading Output Details**

Each output row includes:

* **Risk Description**: The statement being classified.
* **Final Risk Level**: The label assigned (e.g. "low", "medium", "manual review").
* **Final Review Dept**: The assigned department ("electrical", "controls", etc.).
* **Label Source**: Where the label came from:

  * "Rule" (forced by keyword/logic),
  * "Classic" (machine learning, with high confidence),
  * "Similarity (score=...)" (embedding-based match).
* **Similarity Match** (if enabled): The closest training statement.
* **Similarity Score**: Cosine similarity (higher is more similar).

---

## 8. **Manual Review (Critical Step)**

### Why It’s Needed

* This tool is only as good as its data and rules.
* Human domain expertise is always required for final safety and process decisions.

### How To Review

1. Sort or filter output by “Label Source.”
2. Prioritize checking rows labeled as “manual review,” “unclassified,” or with low similarity scores.
3. Confirm/correct the department and risk for every line.
4. If you see repeated mislabels, consider retraining with more/better data or improving rules.

---

## 9. **Troubleshooting & FAQ**

* **No models loaded?**

  * Train or load a model set first.
* **App crashes on file load?**

  * Make sure the file format matches the expected columns and encoding (UTF-8).
* **Model set not appearing?**

  * Click **Refresh Model Sets**.

---

## 10. **Advanced/Power-User Options**

* **Customizing rules:**
  Edit `rules_engine.py` to update department or risk triggers.
* **Changing ML parameters:**
  Update training routines or model hyperparameters in `train_classifier.py` and `classic_ml.py`.
* **Modifying output columns:**
  Edit `logic.py`’s `multipass_classify` function.






**End of README**
