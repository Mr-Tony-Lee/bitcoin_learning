from ALL_Class.Finite_Field import FieldElement
from ALL_Class.Eliptic_curve import Point
from ALL_Class.Helper import encode_base58, hash160 , hash256 , encode_base58_checksum
from ALL_Class.Module import *

P = 2**256 - 2**32 - 977
A = 0 
B = 7
Gx = 0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
Gy = 0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8
N = 0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141

class S256Field(FieldElement):    
    def __init__(self, num , prime = None):
        super().__init__(num=num , prime = P)
    # Now, let's rewrite the square root equation:
    # 𝑤²≡𝑣

    # Since 𝑤ᵖ⁻¹%𝑝≡1, we can multiply both sides by 𝑤ᵖ⁻¹ without changing the equality:
    # 𝑤²≡𝑤²⋅1≡𝑤²⋅𝑤ᵖ⁻¹≡𝑤⁽ᵖ⁺¹⁾

    # Since 𝑝 is odd (recall 𝑝 is prime), dividing (𝑝+1) by 2 will still get an integer (∵ (𝑝+1) is even), implying:
    # 𝑤≡𝑤⁽ᵖ⁺¹⁾ᐟ²

    # Given that the prime 𝑝 satisfies 𝑝%4≡3
    # Means that 𝑝=4𝑘+3, where 𝑘 is some integer
    # Now, add 1 to 𝑝:
    # 𝑝+1=(4𝑘+3)+1=4𝑘+4=4(𝑘+1),
    # Divide both sides by 4:
    # (𝑝+1)/4=𝑘+1
    # Since 𝑘 is an integer, 𝑘+1 is also an integer
    # -> We prove that (p+1)/4 is an integer
    def sqrt(self):
        return self**((P+1) // 4)

class S256Point(Point):
    def __init__(self, x, y, a = None , b = None ):
        a , b = S256Field(A), S256Field(B)
        if type(x) == int and type(y) == int :
            x , y = S256Field(x) ,S256Field(y)
        elif type(x) == int :
            x = S256Field(x) 
        elif type(y) == int :
            y = S256Field(y)
        super().__init__(x = x , y = y , a = a , b = b )
    def __rmul__(self, coefficient):
        coef = coefficient % N 
        return super().__rmul__(coef)
    def verify(self, z , sig ):
        s_inv = pow(sig.s , N-2 , N )
        u = z * s_inv % N 
        v = sig.r * s_inv % N 
        total = u * G + v * self
        return total.x.num == sig.r
    
    '''Convert a 256-bit number into 32 bytes, big-endian'''
    '''
        Use the to_bytes method in Python 3
        在 Python 中使用 to_bytes 方法將數字轉換為 bytes -> num.to_bytes(1st , 2st)
        1st argument: How many bytes it should take up 佔用多少 bytes
        2nd argument: The endianness 大端序還是小端序
    '''
    def sec(self , compressed = True ):
        '''returns the binary version of the SEC format'''

        # # If 𝐴(𝑥, 𝑦) is on the curve, −𝐴(𝑥, −𝑦) is also on the curve
        # In finite field 𝐹𝑝: −𝑦%𝑝≡(𝑝−𝑦)%𝑝
        # Solutions for 𝑥: 𝐴(𝑥, 𝑦) or −𝐴(𝑥, 𝑝−𝑦)
        # 𝑦 and 𝑝−𝑦 are always 1 even and 1 odd , since p is the prime number larger than 2 , p is odd.

        # The 𝑦-coordinate is compressed into a single byte (even (0x02) or odd (0x03))

        #  Uncompressed Format
        #  𝑥-coordinate = ffe5…d57c (32 bytes)
        #  𝑦-coordinate = 315d…1d10 (32 bytes)
        #  -> Prefix 0x04 | ffe5…d57c | 315d…1d10

        #  Compressed Format
        #  𝑦-coordinate is even → prefix 0x02 ->  Prefix 0x02 | ffe5…d57c

        # parity bit -> \x04 -> uncompressed 
        # \x02 -> y is even .
        # \x03 -> y is odd .

        if compressed:
            if self.y.num % 2 == 0 :
                return b'\x02' + self.x.num.to_bytes(32,'big')
            else:
                return b'\x03' + self.x.num.to_bytes(32,'big')
        
        '''Convert a 256-bit number into 32 bytes, big-endian'''
        return b'\x04' + self.x.num.to_bytes(32,'big') + self.y.num.to_bytes(32,'big')
    

    # Write a parse method in the S256Point class to figure out which 𝑦 we need when getting a serialized SEC pubkey:
    def parse(self, sec_bin):
        '''return a Point object from a SEC binary (not hex )'''
        if sec_bin[0] == 4 :    # Is uncompressed SEC format or not ? -> here is not compress -> \x04 x-cord y-cord
            x = int.from_bytes(sec_bin[1:33],'big')
            y = int.from_bytes(sec_bin[33:65], 'big')
            return S256Point(x = x , y = y)
        
        # here is compress -> if even \x02 x-cord , odd \x03 x-cord 
        is_even = sec_bin[0] == 2 # The evenness of the 𝑦-coordinate is given in the first byte (0x02 →2 in decimal) 𝑦座標的奇偶性

        x = S256Field(int.from_bytes(sec_bin[1:] , 'big'))  
        
        # The right side equation y^2 = x^3 + 7 
        alpha = x**3 + S256Field(B) # B = 7 
        beta = alpha.sqrt() # 𝑦≡(𝑥³+7)⁽ᵖ⁺¹⁾ᐟ⁴% p

        # y 一定是 一奇數 一偶數 
        # beta 可能解為 𝑦 and 𝑝 – 𝑦

        if beta.num % 2 == 0 :
            even_beta = beta 
            odd_beta = S256Field(P-beta.num)
        else:
            even_beta = S256Field(P-beta.num)
            odd_beta = beta 

        if is_even:
            return S256Point(x , even_beta)
        else:
            return S256Point(x , odd_beta)
        
    def hash160(self , compressed = True):
        return hash160(self.sec(compressed))
    
    # A unique identifier that represents a destination for Bitcoin payments
    # It is the hash of ECDSA public key (SEC)

    # Compute a bitcoin address:
    # Version = Prefix
    # Public Key Hash (=fingerprint) = RIPEMD160(SHA256(Public Key))
    # Checksum = 1st 4 bytes of SHA256(SHA256(Version concatenated with Public Key Hash))
    # Bitcoin address = Base58Encode( Public Key Hash concatenated with checksum)

    # What is testnet?
    # A parallel Bitcoin network for development and testing
        # The testnet coins are worthless
    # The testnet chain has significantly more blocks than mainnet
        # Testnet has a lower mining difficulty, leading to faster block production
        # The proof-of-work required to find a block is relatively easy
    def address(self , compressed = True , testnet = False):
        '''Returns the addresss string '''
        # Step 2. Hashing:
            # Apply SHA256 followed by the RIPEMD160 to the
            # SEC format. This process is called Hash160
        h160 = self.hash160(compressed) # hashing 

        # Step 1. Prefix:
            # Mainnet addresses: 0x00; Testnet: 0x6f
        if testnet:
            prefix = b'\x6f'
        else:
            prefix = b'\x00'

        # Step 3.
            # Add prefix (0x00 or 0x6f) to the Hash160 result
        return encode_base58_checksum(prefix + h160)

class Signature:
    def __init__(self, r , s ):
        self.r = r 
        self.s = s     
    
    # 0x30| Total Length | 0x02 | Length(𝑟) | 𝑟 | 0x02 | Length(𝑠) | 𝑠
    # ----
    # 0x30 : The header byte indicating a DER type (always 0x30)
    # 0x45( Total Length ) : 1 byte to encode the length of the rest of the signature (𝑟, 𝑠) (usually 0x44 or 0x45)
    # 0x02 : The header byte indicating an integer
    # 0x21(Length(𝑟)) : 1 byte to encode the length of the following 𝑟 value
    # r 
    # 0x02 : The header byte indicating an integer
    # 0x20 : 1 byte to encode the length of the following 𝑠 value
    # s 

    # der rules 
    # Encoding rules : The 𝑟 and 𝑠 value must be prepended with 0x00 if their first byte ≥ 0x80 (2補數，0x80 = 1000 會被電腦判斷為負數，但r,s都是正的)
    # 所以要加 prefix 讓他都變成正數
    # □ E.g., 0xed ≥ 0x80 → 00ed..8f
    # DER signatures have variable lengths, typically 71 bytes on average

    def der(self):
        rbin = self.r.to_bytes(32, byteorder = 'big')
        # remove all null bytes at the beginning 
        rbin = rbin.lstrip(b'\x00') # 切掉最左邊的指定字符，把無效bit給切掉
        
        #if rbin has a high bit , add a \x00 
        if rbin[0] & 0x80:
            rbin = b'\x00' + rbin 
        result = bytes([2, len(rbin)]) + rbin
        # 把list裡面的number轉成bytes -> 0x02 , length(r) + r 
        # 0x02 | Length(𝑟) | 𝑟 

        # 0x02 | Length(𝑠) | 𝑠
        sbin = self.s.to_bytes(32 , byteorder = 'big')
        # remove all null bytes at the beginning 
        sbin = sbin.lstrip(b'\x00')
        
        #if sbin has a high bit add a \x00
        if sbin[0] & 0x80:
            sbin = b'\x00' + sbin 
        result += bytes([2, len(sbin)]) + sbin

        return bytes([0x30, len(result)]) + result 
        #  0x30| Total Length(後6格的內容長度) | 0x02 | Length(𝑟) | 𝑟 | 0x02 | Length(𝑠) | 𝑠
    
    
class PrivateKey:
    def __init__(self,secret):
        self.secret = secret
        self.point = secret * G 
            
    def sign(self, z ):
        k = random.randint(0,N)
        # k = self.deterministic_k(z)
        r = (k*G).x.num
        k_inv = pow(k,N-2,N)
        s = (z + r*self.secret) * k_inv % N 
        if s > N/2:
            s = N-s
        return Signature(r,s)
    
    # Wallet Import Format (WIF) : 
        # A serialization of the private key that’s meant to be human-readable and easy to copy
        # Use the same Base58 encoding that Bitcoin addresses use
    def wif(self , compressed = True , testnet = False):
        
        # Step 2. Serialized secret key:
        #   Encode the private key in 32-byte big-endian
        secret_bytes = self.secret.to_bytes(32,'big')

        # Step 1. Prefix:
            # Mainnet private keys : 0x80 
            # Testnet private keys : 0xef
        if testnet:
            prefix = b'\xef'
        else :
            prefix = b'\x80'
        
        # Step 3. Suffix:
            # If the SEC format used for the public key address was compressed, add a suffix of 0x01
        if compressed:
            suffix = b'\x01'
        else:
            suffix = b''
        
        # Step 4.
            # Combine prefix, serialized secret, and suffix

        # Step 5. Checksum:
            # Perform double SHA256 on the result from Step 4
            # Extract the first 4 bytes as the checksum

        # Step 6. Encode:
            # Combine Step 4 and 5, then encode using Base58

        return encode_base58_checksum(prefix + secret_bytes + suffix )
    
G = S256Point(Gx,Gy)
if __name__ == '__main__':
    print(N*G)
    print(encode_base58(b'10000'))