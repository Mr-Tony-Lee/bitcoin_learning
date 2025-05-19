from Bitcoin_S256Point import * 


print("-------------------------- Problem 1 --------------------------\n")
def problem1():
    Px = 0x801be5a7c4faf73dd1c3f28cebf78d6ba7885ead88879b76ffb815d59056af14
    Py = 0x826ddfcc38dafe6b8d463b609facc009083c8173e21c5fc45b3424964e85f49e
    r_sig = 0xf01d6b9018ab421dd410404cb869072065522bf85734008f105cf385a023a80f 
    s_sig = 0x22afcd685b7c0c8b525c2a52529423fcdff22f69f3e9c175ac9cb3ec08de87d8
    z = 0x90d7aecf3f2855d60026f10faab852562c76e7e043cf243474ba5018447c2c22

    p = S256Point(Px,Py)
    sig = Signature(r_sig , s_sig) 
    
    print(p.verify(z,sig))
problem1()
print("\n-------------------------- Problem 1 --------------------------\n")

print("-------------------------- Problem 2 --------------------------\n")
class PrivateKey_for_2 :
    def __init__(self,secret):
        self.secret = secret
        self.point = secret * G 

    def sign(self, z):
        k = 1234567
        r = (k*G).x.num
        k_inv = pow(k,N-2,N)
        s = (z + r*self.secret) * k_inv % N 
        if s > N/2:
            s = N-s
        return Signature(r,s)

def problem2():
    e = 1234567
    z = int.from_bytes(hash256(b'Introduction to Bitcoin homework 2.2'), 'big') 
    pri = PrivateKey_for_2(secret = e)
    pri_sign = pri.sign(z = z )
    print(f"z = {hex(z)} \nr = {hex(pri_sign.r)} \ns = {hex(pri_sign.s)}")

problem2()
print("\n-------------------------- Problem 2 --------------------------\n")


print("-------------------------- Problem 3-1 --------------------------\n")
def problem3_1():
    e = 23396049
    P = e * G
    print("the uncompressed SEC format when secret is 23396049 :", P.sec(False))

problem3_1()
print("\n-------------------------- Problem 3-1 --------------------------\n")

print("-------------------------- Problem 3-2 --------------------------\n")
def problem3_2():
    e = 23396050
    p = e * G
    print("the compressed SEC format when secret is 23396050 :", p.sec(True))

problem3_2()
print("\n-------------------------- Problem 3-2 --------------------------\n")

print("-------------------------- Problem 3-3 --------------------------\n")

def problem3_3():
    r = 0x8208f5abf04066bad1db9d46f8bcf5a6cc11d0558ab523e7bd3c0ec08bdb782f
    s = 0x22afcd685b7c0c8b525c2a52529423fcdff22f69f3e9c175ac9cb3ec08de87d8
    print(Signature(r,s).der())
    
problem3_3()

print("\n-------------------------- Problem 3-3 --------------------------\n")


print("-------------------------- Problem 4-1 --------------------------\n")

def problem4_1():
    e1 = 23396051
    p1 = PrivateKey(e1).point
    print("When secret key is 23396051, the address corresponding to Public Keys is :", p1.address( False , True ))
    e2 = 23396052
    p2 = PrivateKey(e2).point
    print("When secret key is 23396052, the address corresponding to Public Keys is :", p2.address( True , True ))

problem4_1()

print("\n-------------------------- Problem 4-1 --------------------------\n")

print("-------------------------- Problem 4-2 --------------------------\n")

def problem4_2():
    e = 23396053
    p_key = PrivateKey(e)
    print("the WIF for Private Key whose the secret is 23396053 is :",p_key.wif(True , True ))

problem4_2()

print("\n-------------------------- Problem 4-2 --------------------------\n")