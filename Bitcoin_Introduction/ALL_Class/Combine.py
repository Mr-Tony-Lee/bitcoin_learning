from ALL_Class.Eliptic_curve import Point
from ALL_Class.Finite_Field import FieldElement

prime = 11
a = FieldElement(num = 0 ,prime = prime)
b = FieldElement(num = 2 , prime = prime)
x1 = FieldElement(num = 1 , prime = prime)
y1 = FieldElement(num = 6 , prime = prime)
x2 = FieldElement(num = 6 , prime = prime)
y2 = FieldElement(num = 3 , prime = prime)
p1 = Point(x1,y1,a,b)
p2 = Point(x2,y2,a,b)

print(2*p1)