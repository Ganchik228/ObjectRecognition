R, B = map(int, input().split())

S = (R + 4) // 2
P = B + R

discriminant = S * S - 4 * P
sqrt_disc = int(discriminant ** 0.5)

W = (S + sqrt_disc) // 2
H = (S - sqrt_disc) // 2

print(W, H)
