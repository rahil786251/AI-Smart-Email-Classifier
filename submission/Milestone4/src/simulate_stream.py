import time
from predict_email import predict_email

emails = [
    "My order has not arrived, please help",
    "Win $1000 now click here",
    "Server down urgently fix",
    "Thank you for the great support"
]

for email in emails:
    category, urgency = predict_email(email)
    print("Email:", email)
    print("Category:", category, "| Urgency:", urgency)
    print("-" * 50)
    time.sleep(2)
