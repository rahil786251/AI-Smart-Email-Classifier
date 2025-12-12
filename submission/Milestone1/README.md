AI-Powered Smart Email Classifier — Milestone 1 (Data Collection \& Preprocessing)



Objective: Prepare a cleaned and labeled dataset for model training.



Deliverables in this folder



data/processed/emails\_clean.csv — cleaned emails (HTML/signatures removed, normalized).



data/processed/emails\_labeled.csv — labeled dataset with category and urgency columns.



src/preprocess.py — preprocessing script used to clean the raw emails.



src/run\_cleaner.py — wrapper script to run the cleaner.



LICENSE — MIT License (as provided).



This README.md.



How files were produced



Raw email file placed at data/raw/emails.csv.



Run python src\\run\_cleaner.py which:



detects the text column (body/clean\_text/subject),



removes HTML, quoted replies, signatures,



normalizes whitespace and removes URLs/emails,



outputs data/processed/emails\_clean.csv.



Labeled dataset emails\_labeled.csv was created by adding two columns:



category ∈ {complaint, request, feedback, spam}



urgency ∈ {high, medium, low}

