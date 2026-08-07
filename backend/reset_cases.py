# reset_cases.py, in backend/
from database import update_case
for i in range(1, 8):
    update_case(i, action_taken="none")
print("reset done")