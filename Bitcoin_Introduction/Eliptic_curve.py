class Point:
    def __init__ (self , x , y , a , b):
        self.a = a 
        self.b = b 
        self.x = x
        self.y = y 
        # 若𝑥軸和𝑦軸座標 None，表示該點為無窮遠點
        # return 會直接返回 Point(infinity)
        # 在 Python 中 None 是沒資料，不等於 0
        if self.x is None and self.y is None:   # <1>
            return 
        # 檢查某點是否在橢圓曲線上
        if self.y**2 != self.x**3 + self.x*self.a + b :
            return ValueError('({},{} is not on the curve)'.format(x,y))
    def __repr__(self):
        if(self.x == None and self.y == None):
            return "Point(Infinity)"
        return "Point({},{})".format(self.x , self.y )
    # 兩點相等:若且唯若在相同曲線上以及擁有相同座標時才相等
    def __eq__ (self , other ):
        return self.x == other.x and self.y == other.y and self.a == other.a and self.b == other.b 
    def __ne__ (self,other):
        return not(self == other)
    def __add__ (self , other):
        if self.a != other.a and self.b != other.b: # <2>
            raise TypeError('Point {}, {} are not on the same curve'.format(self,other))
        # If self is the point at infinity, the function returns other, meaning:𝑂 + 𝑃 = 𝑃
        if self.x is None:     # <3>
            return other 
        if other.x is None:    # <4>
            return self
        if self == other and self.y == 0:
            return self.__class__(None , None , self.a , self.b)
        # 𝑦=𝑠(𝑥−𝑥₁)+𝑦₁ (Point-slope equation) (1)
        # 𝑦²=(𝑠(𝑥−𝑥₁)+𝑦₁)²=𝑥³+𝑎𝑥+𝑏  (2)
        # 𝑥³−𝑠²𝑥²+(𝑎+2𝑠²𝑥₁−2𝑠𝑦₁)𝑥+𝑏−𝑠²𝑥₁²+2𝑠𝑥₁𝑦₁−𝑦₁²=0  (3) 全部展開，並移到同一邊
        # Since x1 , x2 , x3 is the solution to this equation , thus 
        # (𝑥−𝑥₁)(𝑥−𝑥₂)(𝑥−𝑥₃)=0 -> 𝑥₃–(𝑥₁+𝑥₂+𝑥₃)𝑥₂+(𝑥₁𝑥₂+𝑥₁𝑥₃+𝑥₂𝑥₃)𝑥–𝑥₁𝑥₂𝑥₃=0
        # from Vieta's formula （根與係數關係）->  the coefficients have to equal eachother if the roots are the same
        # −𝑠²=–(𝑥₁+𝑥₂+𝑥₃) -> 𝑥₃=𝑠²–𝑥₁–𝑥₂
        if self.x != other.x:
            s = (other.y - self.y ) / (other.x - self.x)
        else:
            s = (3*self.x**2 + self.a ) / (2*self.y)
        # 𝑦²=𝑥³+𝑎𝑥+𝑏
        # 2𝑦⋅𝑑𝑦=(3𝑥²+𝑎)𝑑𝑥
        # 𝑑𝑦/𝑑𝑥=(3𝑥²+𝑎)/(2𝑦)
        x = s**2 - self.x - other.x 
        y = s * (self.x - x ) - self.y 
        return self.__class__(x , y , self.a , self.b) 
    def __rmul__(self,  coefficient):
        coef = coefficient
        current = self
        result = self.__class__(None, None , self.a , self.b )
        while(coef):
            if coef & 1 :
                result += current 
            current += current
            coef >>= 1 
        return result 