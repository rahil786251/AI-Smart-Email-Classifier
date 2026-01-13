\# Milestone 3: Urgency Detection \& Scoring



\## Objective

To implement urgency prediction for enterprise emails using a machine learning approach combined with keyword-based logic.



---



\## Dataset

\- Source: Large email dataset (>5000 emails)

\- File used: `data/processed/emails\_with\_urgency.csv`

\- Columns:

&nbsp; - `text`: email content

&nbsp; - `target`: spam / non-spam label

&nbsp; - `urgency`: high / medium / low



---



\## Approach



\### 1. Urgency Label Generation

Urgency labels were generated using a hybrid approach:

\- \*\*Keyword-based rules\*\* (e.g., "urgent", "asap", "immediately")

\- \*\*Spam emails\*\* were automatically assigned low urgency

\- Normal requests were marked as medium urgency



\### 2. Machine Learning Model

\- Algorithm: \*\*Multinomial Naive Bayes\*\*

\- Feature extraction: \*\*TF-IDF Vectorization\*\*

\- Train-test split: 80% training, 20% testing



---



\## Evaluation Metrics

\- Accuracy

\- Precision

\- Recall

\- F1-score

\- Confusion Matrix



---



\## Results

The urgency classification model achieved high accuracy and balanced F1-scores across all urgency classes, validating the effectiveness of the hybrid urgency detection strategy.



---



\## Files Included



