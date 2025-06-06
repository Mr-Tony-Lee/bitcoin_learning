class FieldElement:
    def __init__(self, num , prime):
        if( num >= prime or num < 0 ):
            error = "Num {} not in field range 0 to {}".format(num, prime - 1)
            raise ValueError(error)
        self.num = num 
        self.prime = prime
    def __repr__(self):
        return "FieldElement_{}({})".format(self.prime , self.num )
    def __eq__(self, other):
        if other is None:
            return False
        if(type(other) == int ):
            other = self.__class__(other,self.prime)
        return self.num == other.num and self.prime == other.prime
    def __ne__(self, other):
        return not(self == other)
    def __add__(self, other):
        if self.prime != other.prime:
            raise TypeError("Cannot add two numbers in different Fields")
        num = (self.num + other.num) % self.prime
        return self.__class__(num,self.prime)
    def __sub__(self, other):
        if self.prime != other.prime:
            raise TypeError("Cannot subtract two numbers in different Fields")
        num = (self.num - other.num) % self.prime
        return self.__class__(num,self.prime)
    def __mul__(self , other ):
        if self.prime != other.prime:
            raise TypeError("Cannot multiply two numbers in different Fields")
        num = ((self.num % self.prime) * (other.num % self.prime)) % self.prime
        return self.__class__(num,self.prime)
    def __rmul__(self , coefficient ):
        coef = coefficient
        current = self
        result = self.__class__(0,self.prime)
        while(coef):
            if coef & 1 :
                result += current 
            current += current
            coef >>= 1 
        return result 
    def __pow__(self,exponent):
        n = exponent % (self.prime-1)
        num = pow(self.num, n , self.prime) 
        return self.__class__(num, self.prime)
    def __truediv__(self , other):
        if self.prime != other.prime:
            raise TypeError("Cannot multiply two numbers in different Fields")
        num = ((self.num % self.prime) * pow(other.num ,self.prime-2 , self.prime ) % self.prime ) % self.prime
        return self.__class__(num,self.prime)

if __name__ == "__main__":
    print(FieldElement(6,11)*(FieldElement(3,11)**-1))    
    print(FieldElement(-82549%11,11)*(FieldElement(216%11,11)**-1))
    print(FieldElement(16**9%11,11))
    print(FieldElement(16%11,11)**-1)

    a = FieldElement(337,701)
    b = FieldElement(557,701)
    print((a**79)*(b**131))
    c = FieldElement(31,881)
    d = FieldElement(800,881)
    print(-5%11)
    # print(c**(-1) * d )

    
    