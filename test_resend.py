import os
import resend
from dotenv import load_dotenv

load_dotenv()

resend.api_key = os.getenv("RESEND_API_KEY")
from_email = os.getenv("RESEND_FROM_EMAIL")
to_email = "triptibhatt74@gmail.com"

print("API key loaded:", bool(resend.api_key))
print("From:", from_email)
print("To:", to_email)

try:
    response = resend.Emails.send({
        "from": from_email,
        "to": [to_email],
        "subject": "Gaash OTP test",
        "text": "This is a test email from the Gaash authentication service.",
    })

    print("SUCCESS")
    print("Response:", response)

except Exception as exc:
    print("FAILED")
    print(repr(exc))