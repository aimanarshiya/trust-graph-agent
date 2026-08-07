from database import get_case

for i in range(1, 8):
    case = get_case(i)
    if case:
        print(f"{i}: {case['seller_id']} -> has_explanation={bool(case['explanation'])}")
    else:
        print(f"{i}: not found")