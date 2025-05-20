from Module import *
def op_0(stack):
    stack.append(encode_num(0))    
    return True 


def op_dup(stack):
    """
    Duplicates the top element of the stack.
    """
    if len(stack) < 1 :
        return False
    stack.append(stack[-1])
    return True 

def op_hash160(stack):
    """
    Hashes the top element of the stack using SHA-256 followed by RIPEMD-160.
    """
    if len(stack) < 1:
        return False
    top_element = stack.pop()
    hash_result = hashlib.new('ripemd160', hashlib.sha256(top_element).digest()).digest()
    stack.append(hash_result)
    return True

def op_hash256(stack):
    """
    Hashes the top element of the stack using SHA-256 twice.
    """
    if len(stack) < 1:
        return False
    top_element = stack.pop()
    hash_result = hashlib.sha256(hashlib.sha256(top_element).digest()).digest()
    stack.append(hash_result)
    return True

def op_ripemd160(stack):
    """
    Hashes the top element of the stack using RIPEMD-160.
    """
    if len(stack) < 1:
        return False
    top_element = stack.pop()
    hash_result = hashlib.new('ripemd160', top_element).digest()
    stack.append(hash_result)
    return True

def op_sha256(stack):
    """
    Hashes the top element of the stack using SHA-256.
    """
    if len(stack) < 1:
        return False
    top_element = stack.pop()
    hash_result = hashlib.sha256(top_element).digest()
    stack.append(hash_result)
    return True

def op_checksig(stack):
    """
    Verifies a signature against a public key and message.
    """
    if len(stack) < 3:
        return False
    sig = stack.pop()
    pubkey = stack.pop()
    msg = stack.pop()
    
    # Here you would implement the actual signature verification logic
    # For now, we will just append True to the stack to indicate success
    stack.append(True)
    return True

def op_6(stack):
    stack.append(6)
    return True

def op_equal(stack):
    if len(stack) < 2:
        return False
    a = stack.pop()
    b = stack.pop()
    if a == b:
        stack.append(1)
    else:
        stack.append(0)
    return True

def op_add(stack):
    if len(stack) < 2:
        return False
    a = stack.pop()
    b = stack.pop()
    # 確保 a、b 是整數，如果是 bytes 就轉換
    if isinstance(a, bytes):
        a = int.from_bytes(a, 'big')
    if isinstance(b, bytes):
        b = int.from_bytes(b, 'big')
    stack.append(a + b)
    return True

def op_mul(stack):
    if len(stack) < 2:
        return False
    a = stack.pop()
    b = stack.pop()
    if isinstance(a, bytes):
        a = int.from_bytes(a, 'big')
    if isinstance(b, bytes):
        b = int.from_bytes(b, 'big')
    stack.append(a * b)
    return True


# Complete OP_add
from ecdsa import VerifyingKey, SECP256k1, BadSignatureError
import hashlib

def op_checksig(stack, z):
    if len(stack) < 2:
        return False
    pubkey = stack.pop()
    signature = stack.pop()

    # 移除最後一個字節的 hash type（SIGHASH_ALL）
    signature = signature[:-1]

    try:
        # 建立 VerifyingKey 物件
        vk = VerifyingKey.from_string(pubkey, curve=SECP256k1)

        # 驗證簽章（z 是事先算好的交易 hash）
        if vk.verify(signature, z, hashfunc=hashlib.sha256):
            stack.append(1)
        else:
            stack.append(0)
    except (BadSignatureError, ValueError):
        stack.append(0)

    return True


OP_CODE_FUNCTIONS = {
    0 : op_0,
    86 : op_6, 
    135 : op_equal, 
    147 : op_add, 
    149 : op_mul, 
    118 : op_dup,
    166 : op_ripemd160,
    168 : op_sha256,
    169 : op_hash160,
    170 : op_hash256,
    172 : op_checksig
}

OP_CODE_NAMES = {
    0 : "op_0",
    86 : "op_6", 
    135 : "op_equal", 
    147 : "op_add", 
    149 : "op_mul", 
    118 : "op_dup",
    166 : "op_ripemd160",
    168 : "op_sha256",
    169 : "op_hash160",
    170 : "op_hash256",
    172 : "op_checksig"
}

def encode_num(num):
    if num == 0 :
        return b''
    abs_num = abs(num)
    negative = num < 0 
    result = bytearray()
    while abs_num > 0:
        result.append(abs_num & 0xff)
        abs_num >>= 8
    
    if result[-1] & 0x80:
        if negative:
            result.append(0x80)
        else:
            result.append(0x00)
    elif negative:
        result[-1] |= 0x80
    
    return bytes(result)

def decode_num(element):
    if element == b'':
        return 0 
    big_endian = element[::-1]
    if big_endian[-1] & 0x80:
        negative = True
        big_endian[-1] &= 0x7f
    else:
        negative = False
        result = big_endian[0]
    
    for c in big_endian[1:]:
        result = (result << 8) + c
    
    if negative:
        result = -result
    return result

