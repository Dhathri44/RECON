def additive_increase_multiplicative_decrease(rate_mbps, congestion, min_rate=0.1, max_rate=50.0):
    """Simple AIMD rate controller for experiments, not TCP congestion control."""
    rate = float(rate_mbps)
    if congestion:
        rate *= 0.5
    else:
        rate += 0.1
    return min(max_rate, max(min_rate, rate))
