from ALL_Class.Module import * 

def hash160(s):
    '''sha256 followed by ripemd160'''
    # What is RIPEMD160?
    # A cryptographic hash function based on the Merkle–Damgård construction
    # An improved version of RIPEMD(RACE Integrity Primitives Evaluation Message Digest),
    # increasing output size from 128-bit to 160-bit for better security
    # The development idea of RIPEMD is based on MD4 which in itself is a weak hash function
    
    # digest() return the digest of the data -> 獲取雜湊值 (bytes)
    return hashlib.new('ripemd160' , hashlib.sha256(s).digest()).digest()

BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'

def encode_base58(s):
    BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    # Uses 58 characters : 10 digits + 26 uppercase letters + 26 + lowercase letters + ‘+’ + ‘/’ (in Base64)
    # - 6 confusing characters (0, O, l, I, ‘+’ ,‘/’)
    # Improves read    
    # Base64:
    # Shorter but prone to errors (e.g., 0/O, l/I, + and /)
    # Express 6 bits per character (log₂64=6) vs. Base58: log₂(58)≈5.86
    # Uncompressed SEC: 87 bytes
    # (65 bytes=65×8=520 bits, 520 bits/6=86.6≈87 characters)
    # Compressed SEC: 44 bytes (33 bytes=33×8=264 bits 264 bits/6=44 character) 

    count = 0 
    for c in s :    # The loop is to determine how many of bytes at the front are 0 bytes
        if c == 0 :
            count += 1 
        else:
            break
    num = int.from_bytes(s,'big')
    prefix = '1' * count  # Add them back at the end
    result = ''
    while num > 0 : # The loop that figures out what Base58 digit to use
        num , mod = divmod(num, 58)
        result = BASE58_ALPHABET[mod] + result
        # why cann't result += BASE58_ALPHABET[mod] , it will wrong (reverse)
    return prefix + result 
    
def hash256(s): # double sha256 hashing 
    return hashlib.sha256(hashlib.sha256(s).digest()).digest()
    
def encode_base58_checksum(b):
    # Step 4. Checksum:
        # Perform double SHA256 on the result from Step 3 
        # Extract the first 4 bytes as the checksum
    # Step 5. Encode:
        # Combine Step 3 (Prefix+Hash160) and Step 4 (Checksum), then encode using Base58
    return encode_base58(b + hash256(b)[:4])

def decode_base58(s):
    num = 0 
    BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    for c in s:
        num *= 58
        num += BASE58_ALPHABET.index(c)
    combined = num.to_bytes(25, 'big')
    checksum = combined[-4:]
    if hash256(combined[:-4])[:4] != checksum:
        raise ValueError(f"Invalid Base58 string {s}")
    return combined[1:-4] # remove prefix 0x00

def p2pkh_script(h160):
    from ALL_Class.Script import Script 
    """ Take a hash160 and return the p2pkh ScriptPubKey """
    return Script([0x76, 0xa9 ,h160 ,0x88, 0xac])


def read_varint(s):
    '''Read a variable-length integer from the stream'''
    # 1. 讀取第一個字元
    first_byte = s.read(1)[0]
    # 2. 判斷長度
    if first_byte == 0xfd:
        # 0xfd, 代表後面有兩個位元組
        return int.from_bytes(s.read(2), 'little')
    elif first_byte == 0xfe:
        # 0xfe, 代表後面有四個位元組
        return int.from_bytes(s.read(4), 'little')
    elif first_byte == 0xff:
        # 0xff, 代表後面有八個位元組
        return int.from_bytes(s.read(8), 'little')
    else:
        return first_byte
    
def encode_varint(n):
    '''Encode a variable-length integer'''
    if n < 0xfd:
        return n.to_bytes(1, 'little')
    elif n <= 0xffff:
        return b'\xfd' + int_to_little_endian(n, 2)
    elif n <= 0xffffffff:
        return b'\xfe' + int_to_little_endian(n, 4)
    elif n <= 0xffffffffffffffff:
        return b'\xff' + int_to_little_endian(n, 8)
    else:
        return ValueError('integer too large:{}'.format(n))

def little_endian_to_int(b):
    '''Convert a little-endian byte string to an integer'''
    return int.from_bytes(b, 'little')

def int_to_little_endian(n, length):
    '''Convert an integer to a little-endian byte string of a given length'''
    if n < 0:
        raise ValueError('n must be non-negative')
    return n.to_bytes(length, 'little')

def h160_to_p2pkh_address(h160, testnet=False):
    """Convert a hash160 to a P2PKH address"""
    if testnet:
        prefix = b'\x6f'  # Testnet prefix
    else:
        prefix = b'\x00'  # Mainnet prefix
    return encode_base58_checksum(prefix + h160)  # Add prefix and checksum

def h160_to_p2sh_address(h160, testnet=False):
    """Convert a hash160 to a P2SH address"""
    if testnet:
        prefix = b'\xc4'  # Testnet prefix
    else:
        prefix = b'\x05'  # Mainnet prefix
    return encode_base58_checksum(prefix + h160)  # Add prefix and checksum