# test_one.py, in backend/
from agents.remediation_agent import remediate_case
from database import update_case

update_case(2, action_taken="none")  # reset Case 2 so it can be re-tried
result = remediate_case(2)
print(result)