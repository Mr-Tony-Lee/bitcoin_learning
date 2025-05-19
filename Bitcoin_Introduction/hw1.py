from fractions import Fraction
#question 1-1
print("The answer of question 1-1 : ",(599*607*613)%701)

#question 1-2
print("The answer of question 1-2 : ",(23*223*509*666)%701)

#question 1-3
num = 1 
for i in range(79):
    num *= 337
    num %= 701
for j in range(131):
    num *= 557
    num %= 701
print("The answer of question 1-3 : ", num)

#quesition 2-1
num = 1 
# 31^-1 % 881 = 31^881-2 % 881 
for i in range(881-1-1): 
    num *= 31 
    num %= 881
num *= 800
print("The answer of question 2-1 : ", num % 881)

# quesition 2-2
num = 1 
for i in range(881-101-1):
    num *= 201
    num %= 881
num *= 57
print("The answer of question 2-2 : ", num % 881)
 
# function for elliptic curve
def add_elliptic_curve(xp, yp, xq, yq, a):
    if(xp == xq and yq == yp ):
        LAMBDA = Fraction(3*(xp**2) + a, 2*yp)
    else:
        LAMBDA = Fraction(yq-yp,xq-xp)
    xr = LAMBDA**2 - xp - xq
    yr = LAMBDA * (xp - xr) - yp
    return LAMBDA, xr, yr

# slope, xr, yr = add_elliptic_curve(1, 0, 1, 0, -5)
# print("The answer of question 3-0 : " ,f"slope = : {slope}\t, sum of points = ({xr}, {yr})")
# quesition 3-1
slope, xr, yr = add_elliptic_curve(-2, -5, 5, -9, -11)
print("The answer of question 3-1 : " ,f"slope = : {slope}\t, sum of points = ({xr}, {yr})")

# quesition 3-2
slope, xr, yr = add_elliptic_curve(-2, -3, 3, 3, -7)
print("The answer of question 3-2 : ",f"slope = : {slope}\t, sum of points = ({xr}, {yr})")

# quesition 3-3
slope, xr, yr = add_elliptic_curve(-3, -1, 5, 9, -9)
print("The answer of question 3-3 : ",f"slope = : {slope}\t, sum of points = ({xr}, {yr})")

# question 4-1
slope, xr, yr = add_elliptic_curve(5, -9, 5, -9, -11)
print("The answer of question 4-1 : ",f"slope = : {slope}\t, sum of points = ({xr}, {yr})")

# question 4-2
slope, xr, yr = add_elliptic_curve(5, 9, 5, 9, -9)
print("The answer of question 4-2 : ",f"slope = : {slope}\t, sum of points = ({xr}, {yr})")
