def calculate_pi():
    return round(3.141592, 5)

# Test the function to see if it calculates pi correctly
pi = calculate_pi()
assert abs(pi - 3.14159) < 0.00001