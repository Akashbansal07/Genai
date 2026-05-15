from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class InstagramUser(BaseModel):
    name: str
    dob: YYYY-MM-DD
    username : str
    password: str
    phonenumber: int
    email: Optional[str]

User = InstagramUser(name="Hussain", dob=1997-11-18, username="Hussain07", password="12345678", phonenumber=9999999999)

print(User.phonenumber)


