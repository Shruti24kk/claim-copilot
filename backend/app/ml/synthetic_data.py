import random

def generate(n=500):
    texts, labels = [], []
    for _ in range(n):
        if random.random() < 0.35:
            texts.append("urgent payout total loss no witnesses cash only")
            labels.append(1)
        else:
            texts.append("rear ended minor scratch bumper repair estimate")
            labels.append(0)
    return texts, labels
