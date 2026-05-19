from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from datetime import date

load_dotenv()


class InstagramUser(BaseModel):
    name: str
    dob: date
    username: str
    password: str
    phonenumber: int
    email: Optional[str] = None


User = InstagramUser(
    name="Hussain",
    dob="1997-11-18",
    username="Hussain07",
    password="12345678",
    phonenumber=9999999999
)

print(User.phonenumber)