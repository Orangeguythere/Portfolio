from pydantic import BaseModel

class NBAPlayer(BaseModel):
  Game_Played: int
  Minutes_Played: int
  Points: int